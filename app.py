"""
S3 Model Offloader — Flask backend
Scans local model directories, uploads to S3 with path metadata for 1-click restore.
"""

import os
import threading
import datetime
import hashlib
import re
import json
import argparse
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    ProfileNotFound,
)
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="S3 Model Offloader")
parser.add_argument("--port", type=int, default=8900, help="Port to run the server on")
args = parser.parse_args()

app = Flask(__name__, static_folder="static")

# --- Config ---
MODEL_EXTENSIONS = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
    ".ggml",
    ".pkl",
    ".q4_0",
    ".q8_0",
}
CONFIG_FILE = os.getenv("CONFIG_FILE", "settings.json")


def _default_settings() -> dict:
    return {
        "models_root": os.getenv("MODELS_ROOT", "~/models"),
        "s3_bucket": os.getenv("S3_BUCKET", ""),
        "s3_prefix": os.getenv("S3_PREFIX", "models-offload/"),
        "aws_profile": os.getenv("AWS_PROFILE", None),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", None),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", None),
        "aws_session_token": os.getenv("AWS_SESSION_TOKEN", None),
        "include_personal_stuff": env_bool("INCLUDE_PERSONAL_STUFF", False),
        "personal_paths": parse_personal_paths(
            os.getenv("PERSONAL_PATHS", ",".join(DEFAULT_PERSONAL_PATHS))
        ),
    }


def _normalize_settings(raw: dict) -> dict:
    merged = {**_default_settings(), **(raw or {})}
    merged["models_root"] = os.path.expanduser(
        str(merged.get("models_root") or "~/models")
    )
    merged["s3_bucket"] = str(merged.get("s3_bucket") or "")
    merged["s3_prefix"] = str(merged.get("s3_prefix") or "")
    merged["aws_profile"] = str(merged.get("aws_profile") or "").strip() or None
    merged["aws_access_key_id"] = (
        str(merged.get("aws_access_key_id") or "").strip() or None
    )
    merged["aws_secret_access_key"] = (
        str(merged.get("aws_secret_access_key") or "").strip() or None
    )
    merged["aws_session_token"] = (
        str(merged.get("aws_session_token") or "").strip() or None
    )
    merged["include_personal_stuff"] = bool(merged.get("include_personal_stuff", False))
    merged["personal_paths"] = [
        os.path.expanduser(str(p).strip())
        for p in (merged.get("personal_paths") or [])
        if str(p).strip()
    ]
    return merged


def load_settings() -> dict:
    path = Path(CONFIG_FILE)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return _normalize_settings(json.load(f))
        except Exception as e:
            print(f"⚠️ Failed to read {CONFIG_FILE}: {e}. Using defaults.", flush=True)

    settings = _normalize_settings({})
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to write default {CONFIG_FILE}: {e}", flush=True)
    return settings


def save_settings(settings: dict):
    path = Path(CONFIG_FILE)
    normalized = _normalize_settings(settings)
    with path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_personal_paths(raw: str):
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def parse_csv(raw: str):
    if not raw:
        return []
    return [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]


DEFAULT_PERSONAL_PATHS = [
    "/workspace/ComfyUI/custom_nodes",
    "/workspace/ComfyUI/user",
    "/workspacecomfyui_S3_offloader",
    "/workspace/medo_start.sh",
]

_SETTINGS = load_settings()
MODELS_ROOT = _SETTINGS["models_root"]
S3_BUCKET = _SETTINGS["s3_bucket"]
S3_PREFIX = _SETTINGS["s3_prefix"]
AWS_PROFILE = _SETTINGS["aws_profile"]
AWS_ACCESS_KEY_ID = _SETTINGS["aws_access_key_id"]
AWS_SECRET_ACCESS_KEY = _SETTINGS["aws_secret_access_key"]
AWS_SESSION_TOKEN = _SETTINGS["aws_session_token"]
INCLUDE_PERSONAL_STUFF = _SETTINGS["include_personal_stuff"]
PERSONAL_PATHS = _SETTINGS["personal_paths"]
SCAN_EXCLUDE_DIRS = set(
    p.lower()
    for p in parse_csv(
        os.getenv(
            "SCAN_EXCLUDE_DIRS",
            ".git,__pycache__,.venv,venv,node_modules,.cache,.mypy_cache,.pytest_cache",
        )
    )
)

# In-memory progress store — keyed by job_id
jobs = {}

# In-memory log store
logs = []
MAX_LOGS = 500

# In-memory scan caches (to avoid expensive full rescan on every UI refresh)
FILES_CACHE_TTL_SECONDS = int(os.getenv("FILES_CACHE_TTL_SECONDS", "20"))
S3_KEYS_CACHE_TTL_SECONDS = int(os.getenv("S3_KEYS_CACHE_TTL_SECONDS", "30"))

cache_lock = threading.Lock()
scan_lock = threading.Lock()
scan_in_progress = False

files_cache = {
    "cache_key": None,  # (MODELS_ROOT, S3_BUCKET, S3_PREFIX, INCLUDE_PERSONAL_STUFF, PERSONAL_PATHS)
    "tree": None,
    "scanned_at": None,
}

s3_keys_cache = {
    "cache_key": None,  # (S3_BUCKET, S3_PREFIX)
    "keys": set(),
    "fetched_at": None,
}


def add_log(level: str, message: str):
    """Append a log entry. level: info | success | error | warning"""
    entry = {
        "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,
        "msg": message,
    }
    logs.append(entry)
    if len(logs) > MAX_LOGS:
        logs.pop(0)
    # Also print to CLI
    prefix = {"info": "ℹ️ ", "success": "✅", "error": "❌", "warning": "⚠️ "}.get(
        level, "  "
    )
    print(f"[{entry['ts']}] {prefix} {message}", flush=True)


def get_s3_client():
    access_key = str(AWS_ACCESS_KEY_ID or "").strip()
    secret_key = str(AWS_SECRET_ACCESS_KEY or "").strip()
    session_token = str(AWS_SESSION_TOKEN or "").strip() or None
    profile = str(AWS_PROFILE or "").strip() or None

    if access_key and secret_key:
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
        )
    else:
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3")


def is_model_file(path: Path) -> bool:
    return path.suffix.lower() in MODEL_EXTENSIONS


def path_slug(path_str: str) -> str:
    expanded = os.path.expanduser(path_str)
    name = Path(expanded).name or "root"
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-") or "path"
    digest = hashlib.sha1(expanded.encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}-{digest}"


def get_sources():
    sources = [
        {
            "type": "models",
            "label": "Models",
            "root": MODELS_ROOT,
            "key_prefix": "models",
        }
    ]
    if INCLUDE_PERSONAL_STUFF:
        for raw in PERSONAL_PATHS:
            expanded = os.path.expanduser(raw)
            sources.append(
                {
                    "type": "personal",
                    "label": f"Personal · {Path(expanded).name or expanded}",
                    "root": expanded,
                    "key_prefix": f"personal/{path_slug(raw)}",
                }
            )
    return sources


def resolve_source_for_local_path(local_path: str):
    p = Path(local_path).resolve()
    for source in get_sources():
        root = Path(source["root"]).resolve()
        if root.is_file() and p == root:
            return source
        if root.is_dir():
            try:
                p.relative_to(root)
                return source
            except Exception:
                continue
    return None


def source_rel_path(local_path: str, source: dict) -> str:
    root = Path(source["root"]).resolve()
    local = Path(local_path).resolve()
    if root.is_file():
        return root.name
    return str(local.relative_to(root)).replace("\\", "/")


def get_s3_key(local_path: str) -> str:
    """S3 key = prefix + source-rooted path. Restore uses this path to place files back."""
    source = resolve_source_for_local_path(local_path)
    if source is None:
        rel = os.path.relpath(local_path, MODELS_ROOT).replace("\\", "/")
        return f"{S3_PREFIX}{rel}"
    rel = source_rel_path(local_path, source)
    return f"{S3_PREFIX}{source['key_prefix']}/{rel}"


def local_path_from_s3_key(key: str) -> str:
    rel = key[len(S3_PREFIX) :] if key.startswith(S3_PREFIX) else key
    parts = rel.split("/")

    # New format for model files: models/<rel_path>
    if len(parts) >= 2 and parts[0] == "models":
        return os.path.join(MODELS_ROOT, "/".join(parts[1:]))

    # New format for personal files: personal/<slug>/<rel_path>
    if len(parts) >= 3 and parts[0] == "personal":
        slug = parts[1]
        tail = "/".join(parts[2:])
        for raw in PERSONAL_PATHS:
            expanded = os.path.expanduser(raw)
            if path_slug(raw) != slug:
                continue
            root = Path(expanded)
            if root.is_file():
                if not tail or tail == root.name:
                    return str(root)
                return str(root.parent / tail)
            return str(root / tail)

    # Backward compatibility (legacy keys at prefix root)
    return os.path.join(MODELS_ROOT, rel)


def format_size(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def invalidate_scan_caches():
    """Invalidate local/s3 scan caches (used when config changes)."""
    with cache_lock:
        files_cache["cache_key"] = None
        files_cache["tree"] = None
        files_cache["scanned_at"] = None
        s3_keys_cache["cache_key"] = None
        s3_keys_cache["keys"] = set()
        s3_keys_cache["fetched_at"] = None


def get_s3_keys_cached(force_refresh: bool = False):
    if not S3_BUCKET:
        return set()

    now = datetime.datetime.now()
    cache_key = (S3_BUCKET, S3_PREFIX)

    with cache_lock:
        cached_key = s3_keys_cache["cache_key"]
        fetched_at = s3_keys_cache["fetched_at"]
        if (
            not force_refresh
            and cached_key == cache_key
            and fetched_at
            and (now - fetched_at).total_seconds() < S3_KEYS_CACHE_TTL_SECONDS
        ):
            return set(s3_keys_cache["keys"])

    keys = set()
    try:
        s3 = get_s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
            for obj in page.get("Contents", []):
                keys.add(obj["Key"])
    except Exception:
        # Keep serving with stale/empty keys if S3 lookup fails
        with cache_lock:
            if s3_keys_cache["cache_key"] == cache_key:
                return set(s3_keys_cache["keys"])
        return set()

    with cache_lock:
        s3_keys_cache["cache_key"] = cache_key
        s3_keys_cache["keys"] = keys
        s3_keys_cache["fetched_at"] = now
    return keys


def build_files_tree(path: Path, rel_root: Path, s3_keys: set, file_filter):
    node = {
        "name": path.name,
        "path": str(path),
        "rel_path": str(path.relative_to(rel_root)),
        "type": "dir" if path.is_dir() else "file",
    }
    if path.is_dir():
        children = []
        try:
            for child in sorted(
                path.iterdir(), key=lambda x: (x.is_file(), x.name.lower())
            ):
                if child.is_symlink():
                    continue
                if child.is_dir() or file_filter(child):
                    if child.is_dir() and child.name.lower() in SCAN_EXCLUDE_DIRS:
                        continue
                    children.append(
                        build_files_tree(child, rel_root, s3_keys, file_filter)
                    )
        except (PermissionError, OSError):
            pass
        node["children"] = children
        node["file_count"] = sum(1 for c in children if c["type"] == "file")
    else:
        stat = path.stat()
        s3_key = get_s3_key(str(path))
        node["size"] = stat.st_size
        node["size_human"] = format_size(stat.st_size)
        node["s3_key"] = s3_key
        node["on_s3"] = s3_key in s3_keys
    return node


def scan_files_tree(force_s3_refresh: bool = False):
    s3_keys = get_s3_keys_cached(force_refresh=force_s3_refresh)
    source_trees = []

    for source in get_sources():
        root = Path(source["root"])
        if not root.exists():
            add_log("warning", f"Skipping missing source: {source['root']}")
            continue

        file_filter = is_model_file if source["type"] == "models" else (lambda _p: True)

        if root.is_file():
            s3_key = get_s3_key(str(root))
            stat = root.stat()
            source_trees.append(
                {
                    "name": source["label"],
                    "path": f"__source__:{source['key_prefix']}",
                    "rel_path": source["label"],
                    "type": "dir",
                    "children": [
                        {
                            "name": root.name,
                            "path": str(root),
                            "rel_path": root.name,
                            "type": "file",
                            "size": stat.st_size,
                            "size_human": format_size(stat.st_size),
                            "s3_key": s3_key,
                            "on_s3": s3_key in s3_keys,
                        }
                    ],
                    "file_count": 1,
                }
            )
            continue

        tree = build_files_tree(root, root, s3_keys, file_filter)
        tree["name"] = source["label"]
        tree["path"] = f"__source__:{source['key_prefix']}"
        source_trees.append(tree)

    if not source_trees:
        raise FileNotFoundError(
            f"No configured source exists. Checked MODELS_ROOT={MODELS_ROOT}"
        )

    return {
        "name": "Sources",
        "path": "__virtual_root__",
        "rel_path": ".",
        "type": "dir",
        "children": source_trees,
        "file_count": sum(c.get("file_count", 0) for c in source_trees),
    }


def refresh_files_cache(force_s3_refresh: bool = False):
    global scan_in_progress
    with scan_lock:
        with cache_lock:
            scan_in_progress = True
        try:
            tree = scan_files_tree(force_s3_refresh=force_s3_refresh)
            with cache_lock:
                files_cache["cache_key"] = (
                    MODELS_ROOT,
                    S3_BUCKET,
                    S3_PREFIX,
                    INCLUDE_PERSONAL_STUFF,
                    tuple(PERSONAL_PATHS),
                )
                files_cache["tree"] = tree
                files_cache["scanned_at"] = datetime.datetime.now()
        except Exception as e:
            add_log("warning", f"Background scan failed: {e}")
        finally:
            with cache_lock:
                scan_in_progress = False


def trigger_background_refresh(force_s3_refresh: bool = False):
    with cache_lock:
        if scan_in_progress:
            return
    threading.Thread(
        target=refresh_files_cache,
        kwargs={"force_s3_refresh": force_s3_refresh},
        daemon=True,
    ).start()


# --- API ---


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/config")
def get_config():
    return jsonify(
        {
            "models_root": MODELS_ROOT,
            "s3_bucket": S3_BUCKET,
            "s3_prefix": S3_PREFIX,
            "aws_profile": AWS_PROFILE or "",
            "aws_access_key_id": AWS_ACCESS_KEY_ID or "",
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY or "",
            "aws_session_token": AWS_SESSION_TOKEN or "",
            "include_personal_stuff": INCLUDE_PERSONAL_STUFF,
            "personal_paths": PERSONAL_PATHS,
        }
    )


@app.route("/api/config", methods=["POST"])
def update_config():
    global MODELS_ROOT, S3_BUCKET, S3_PREFIX, AWS_PROFILE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, INCLUDE_PERSONAL_STUFF, PERSONAL_PATHS
    d = request.json or {}
    if "models_root" in d:
        MODELS_ROOT = os.path.expanduser(d["models_root"])
    if "s3_bucket" in d:
        S3_BUCKET = d["s3_bucket"]
    if "s3_prefix" in d:
        S3_PREFIX = d["s3_prefix"]
    if "aws_profile" in d:
        AWS_PROFILE = str(d["aws_profile"] or "").strip() or None
    if "aws_access_key_id" in d:
        AWS_ACCESS_KEY_ID = str(d["aws_access_key_id"] or "").strip() or None
    if "aws_secret_access_key" in d:
        AWS_SECRET_ACCESS_KEY = str(d["aws_secret_access_key"] or "").strip() or None
    if "aws_session_token" in d:
        AWS_SESSION_TOKEN = str(d["aws_session_token"] or "").strip() or None
    if "include_personal_stuff" in d:
        INCLUDE_PERSONAL_STUFF = bool(d["include_personal_stuff"])
    if "personal_paths" in d:
        PERSONAL_PATHS = [
            os.path.expanduser(str(p).strip())
            for p in d["personal_paths"]
            if str(p).strip()
        ]
    try:
        save_settings(
            {
                "models_root": MODELS_ROOT,
                "s3_bucket": S3_BUCKET,
                "s3_prefix": S3_PREFIX,
                "aws_profile": AWS_PROFILE,
                "aws_access_key_id": AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
                "aws_session_token": AWS_SESSION_TOKEN,
                "include_personal_stuff": INCLUDE_PERSONAL_STUFF,
                "personal_paths": PERSONAL_PATHS,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Failed to save settings: {e}"}), 500

    invalidate_scan_caches()
    return jsonify({"status": "ok"})


@app.route("/api/files")
def list_files():
    cache_key = (
        MODELS_ROOT,
        S3_BUCKET,
        S3_PREFIX,
        INCLUDE_PERSONAL_STUFF,
        tuple(PERSONAL_PATHS),
    )
    now = datetime.datetime.now()

    with cache_lock:
        cached_tree = files_cache["tree"]
        cached_key = files_cache["cache_key"]
        scanned_at = files_cache["scanned_at"]
        is_scanning = scan_in_progress

    if cached_tree is not None and cached_key == cache_key:
        if (
            scanned_at
            and (now - scanned_at).total_seconds() >= FILES_CACHE_TTL_SECONDS
            and not is_scanning
        ):
            trigger_background_refresh(force_s3_refresh=False)
        return jsonify(cached_tree)

    try:
        tree = scan_files_tree(force_s3_refresh=False)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    with cache_lock:
        files_cache["cache_key"] = cache_key
        files_cache["tree"] = tree
        files_cache["scanned_at"] = now
    return jsonify(tree)


@app.route("/api/s3/list")
def list_s3():
    if not S3_BUCKET:
        return jsonify({"error": "No S3 bucket configured"}), 400
    try:
        s3 = get_s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        files = []
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(S3_PREFIX) :]
                local_path = local_path_from_s3_key(key)
                files.append(
                    {
                        "s3_key": key,
                        "rel_path": rel,
                        "local_path": local_path,
                        "size": obj["Size"],
                        "size_human": format_size(obj["Size"]),
                        "local_exists": os.path.exists(local_path),
                        "last_modified": obj["LastModified"].isoformat(),
                    }
                )
        return jsonify(files)
    except NoCredentialsError:
        return jsonify({"error": "AWS credentials not found"}), 401
    except ProfileNotFound as e:
        return jsonify({"error": f"AWS profile error: {e}"}), 400
    except ClientError as e:
        return jsonify({"error": str(e)}), 500
    except BotoCoreError as e:
        return jsonify({"error": f"AWS SDK error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected server error: {e}"}), 500


# --- Upload ---


@app.route("/api/upload", methods=["POST"])
def upload_files():
    data = request.json
    paths = data.get("paths", [])
    job_id = data.get("job_id")
    if not S3_BUCKET:
        return jsonify({"error": "No S3 bucket configured"}), 400

    total_bytes = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    jobs[job_id] = {
        "total_files": len(paths),
        "done_files": 0,
        "total_bytes": total_bytes,
        "transferred_bytes": 0,
        "current": "",
        "errors": [],
        "finished": False,
    }

    add_log("info", f"Upload started — {len(paths)} file(s)")

    def do_upload():
        s3 = get_s3_client()
        job = jobs[job_id]
        for path in paths:
            job["current"] = os.path.basename(path)
            try:
                s3_key = get_s3_key(path)

                # FIX: use make_callback to properly capture job ref in closure
                def make_callback(j):
                    def cb(n):
                        j["transferred_bytes"] += n

                    return cb

                s3.upload_file(path, S3_BUCKET, s3_key, Callback=make_callback(job))
                job["done_files"] += 1
                add_log("success", f"Uploaded: {os.path.basename(path)}")
            except Exception as e:
                job["errors"].append({"path": path, "error": str(e)})
                job["done_files"] += 1
                add_log("error", f"Upload failed {os.path.basename(path)}: {e}")
        errs = len(job["errors"])
        add_log(
            "info", f"Upload done — {job['done_files'] - errs} succeeded, {errs} errors"
        )
        job["finished"] = True
        trigger_background_refresh(force_s3_refresh=True)

    threading.Thread(target=do_upload, daemon=True).start()
    return jsonify({"job_id": job_id})


# --- Restore ---


@app.route("/api/restore", methods=["POST"])
def restore_files():
    data = request.json
    keys = data.get("keys", [])
    job_id = data.get("job_id")
    if not S3_BUCKET:
        return jsonify({"error": "No S3 bucket configured"}), 400

    # Fetch total bytes upfront for accurate progress
    s3c = get_s3_client()
    total_bytes = 0
    for key in keys:
        try:
            total_bytes += s3c.head_object(Bucket=S3_BUCKET, Key=key)["ContentLength"]
        except Exception:
            pass

    jobs[job_id] = {
        "total_files": len(keys),
        "done_files": 0,
        "total_bytes": total_bytes,
        "transferred_bytes": 0,
        "current": "",
        "errors": [],
        "finished": False,
    }

    add_log("info", f"Restore started — {len(keys)} file(s)")

    def do_restore():
        s3 = get_s3_client()
        job = jobs[job_id]
        for key in keys:
            local_path = local_path_from_s3_key(key)
            job["current"] = os.path.basename(local_path)
            try:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                def make_callback(j):
                    def cb(n):
                        j["transferred_bytes"] += n

                    return cb

                s3.download_file(
                    S3_BUCKET, key, local_path, Callback=make_callback(job)
                )
                job["done_files"] += 1
                add_log("success", f"Restored: {os.path.basename(local_path)}")
            except Exception as e:
                job["errors"].append({"key": key, "error": str(e)})
                job["done_files"] += 1
                add_log("error", f"Restore failed {os.path.basename(local_path)}: {e}")
        errs = len(job["errors"])
        add_log(
            "info",
            f"Restore done — {job['done_files'] - errs} succeeded, {errs} errors",
        )
        job["finished"] = True
        trigger_background_refresh(force_s3_refresh=True)

    threading.Thread(target=do_restore, daemon=True).start()
    return jsonify({"job_id": job_id})


# --- Progress (unified endpoint) ---


@app.route("/api/progress/<job_id>")
def get_progress(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    pct = (
        int(job["transferred_bytes"] / job["total_bytes"] * 100)
        if job["total_bytes"] > 0
        else (
            int(job["done_files"] / job["total_files"] * 100)
            if job["total_files"] > 0
            else 0
        )
    )
    return jsonify(
        {
            "pct": (
                min(pct, 99) if not job["finished"] else 100
            ),  # Don't show 100% until actually done
            "done_files": job["done_files"],
            "total_files": job["total_files"],
            "current": job["current"],
            "errors": job["errors"],
            "finished": job["finished"],
        }
    )


# --- Delete local files ---


@app.route("/api/delete_local", methods=["POST"])
def delete_local():
    """Hard-delete local files. Only call this after confirming S3 upload succeeded."""
    paths = request.json.get("paths", [])
    print(f"[delete_local] received {len(paths)} path(s): {paths}", flush=True)
    deleted, errors = [], []
    for path in paths:
        try:
            os.remove(path)
            deleted.append(path)
            add_log("success", f"Deleted local: {os.path.basename(path)}")
        except Exception as e:
            errors.append({"path": path, "error": str(e)})
            add_log("error", f"Failed to delete local {os.path.basename(path)}: {e}")
    add_log("info", f"Delete local — {len(deleted)} deleted, {len(errors)} errors")
    trigger_background_refresh(force_s3_refresh=False)
    return jsonify(
        {"deleted": len(deleted), "deleted_paths": deleted, "errors": errors}
    )


# --- Delete from S3 ---


@app.route("/api/delete_s3", methods=["POST"])
def delete_s3():
    """Delete files from S3 bucket."""
    keys = request.json.get("keys", [])
    print(f"[delete_s3] received {len(keys)} key(s): {keys}", flush=True)
    if not S3_BUCKET:
        return jsonify({"error": "No S3 bucket configured"}), 400
    s3 = get_s3_client()
    deleted, errors = 0, []
    for key in keys:
        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=key)
            deleted += 1
            add_log("success", f"Deleted from S3: {key.split('/')[-1]}")
        except Exception as e:
            errors.append({"key": key, "error": str(e)})
            add_log("error", f"Failed to delete S3 {key.split('/')[-1]}: {e}")
    add_log("info", f"Delete S3 — {deleted} deleted, {len(errors)} errors")
    trigger_background_refresh(force_s3_refresh=True)
    return jsonify({"deleted": deleted, "errors": errors})


# --- Logs ---


@app.route("/api/logs")
def get_logs():
    return jsonify(list(reversed(logs)))  # most recent first


@app.route("/api/logs", methods=["DELETE"])
def clear_logs():
    logs.clear()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print(f"🚀 S3 Offloader → http://localhost:{args.port}")
    print(f"⚙️ Settings file: {CONFIG_FILE}")
    print(f"📁 Models root : {MODELS_ROOT}")
    print(f"🪣 S3 bucket   : {S3_BUCKET or '(not set)'}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
