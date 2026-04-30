"""
S3 Model Offloader — Flask backend
Scans local model directories, uploads to S3 with path metadata for 1-click restore.
"""

import os
import shutil
import threading
import datetime
import hashlib
import re
import json
import argparse
import sys
import time
import signal
import subprocess
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
        "s3_endpoint_url": os.getenv("S3_ENDPOINT_URL", ""),
        "aws_profile": os.getenv("AWS_PROFILE", None),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", None),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", None),
        "aws_session_token": os.getenv("AWS_SESSION_TOKEN", None),
        "include_personal_stuff": env_bool("INCLUDE_PERSONAL_STUFF", False),
        "personal_paths": parse_personal_paths(
            os.getenv("PERSONAL_PATHS", ",".join(DEFAULT_PERSONAL_PATHS))
        ),
        "comfyui_base": os.getenv("COMFYUI_BASE", "/workspace/ComfyUI"),
        "comfyui_username": os.getenv("COMFYUI_USERNAME", "default"),
        "sync_folders": ["input", "output", "user"],
        "platform_name": os.getenv("PLATFORM_NAME", ""),
    }


def _normalize_settings(raw: dict) -> dict:
    merged = {**_default_settings(), **(raw or {})}
    merged["models_root"] = os.path.expanduser(
        str(merged.get("models_root") or "~/models")
    )
    merged["s3_bucket"] = str(merged.get("s3_bucket") or "")
    merged["s3_prefix"] = str(merged.get("s3_prefix") or "")
    merged["s3_endpoint_url"] = str(merged.get("s3_endpoint_url") or "").strip()
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
    merged["comfyui_base"] = os.path.expanduser(
        str(merged.get("comfyui_base") or "/workspace/ComfyUI")
    )
    merged["comfyui_username"] = str(merged.get("comfyui_username") or "default").strip() or "default"
    merged["sync_folders"] = [
        str(f).strip()
        for f in (merged.get("sync_folders") or ["input", "output", "user"])
        if str(f).strip()
    ]
    merged["platform_name"] = str(merged.get("platform_name") or "").strip()
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
S3_ENDPOINT_URL = _SETTINGS["s3_endpoint_url"]
AWS_PROFILE = _SETTINGS["aws_profile"]
AWS_ACCESS_KEY_ID = _SETTINGS["aws_access_key_id"]
AWS_SECRET_ACCESS_KEY = _SETTINGS["aws_secret_access_key"]
AWS_SESSION_TOKEN = _SETTINGS["aws_session_token"]
INCLUDE_PERSONAL_STUFF = _SETTINGS["include_personal_stuff"]
PERSONAL_PATHS = _SETTINGS["personal_paths"]
COMFYUI_BASE = _SETTINGS["comfyui_base"]
COMFYUI_USERNAME = _SETTINGS["comfyui_username"]
SYNC_FOLDERS = _SETTINGS["sync_folders"]
PLATFORM_NAME = _SETTINGS["platform_name"]
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
    endpoint_url = str(S3_ENDPOINT_URL or "").strip() or None

    # Guard against empty profile env vars (e.g. AWS_PROFILE="") which
    # botocore treats as an explicit (invalid) profile name.
    for env_name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        if str(os.environ.get(env_name, "")).strip() == "":
            os.environ.pop(env_name, None)

    if access_key and secret_key:
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
        )
    else:
        if profile:
            session = boto3.Session(profile_name=profile)
        else:
            session = boto3.Session()

    client_kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    # Cloudflare R2 and some S3-compatible stores require path-style addressing
    if endpoint_url:
        client_kwargs["config"] = boto3.session.Config(s3={"addressing_style": "path"})
    return session.client("s3", **client_kwargs)


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
                # Keep symlinked files (common in ComfyUI model setups),
                # but skip symlinked directories to avoid recursion issues.
                if child.is_symlink() and child.is_dir():
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


@app.route("/api/disk")
def get_disk():
    # Walk up from models_root to find an existing ancestor path
    p = os.path.expanduser(MODELS_ROOT or "")
    while p and p != os.path.dirname(p):
        if os.path.exists(p):
            break
        p = os.path.dirname(p)
    # If we landed on / or found nothing, prefer /workspace when it's a separate mount
    if not p or p == "/":
        try:
            if os.path.exists("/workspace") and os.stat("/workspace").st_dev != os.stat("/").st_dev:
                p = "/workspace"
            else:
                p = "/"
        except OSError:
            p = "/"
    usage = shutil.disk_usage(p)
    pct_used = round(usage.used / usage.total * 100, 1) if usage.total else 0
    return jsonify({"total": usage.total, "used": usage.used, "free": usage.free, "pct_used": pct_used})


@app.route("/api/config")
def get_config():
    return jsonify(
        {
            "models_root": MODELS_ROOT,
            "s3_bucket": S3_BUCKET,
            "s3_prefix": S3_PREFIX,
            "s3_endpoint_url": S3_ENDPOINT_URL or "",
            "aws_profile": AWS_PROFILE or "",
            "aws_access_key_id": AWS_ACCESS_KEY_ID or "",
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY or "",
            "aws_session_token": AWS_SESSION_TOKEN or "",
            "include_personal_stuff": INCLUDE_PERSONAL_STUFF,
            "personal_paths": PERSONAL_PATHS,
            "comfyui_base": COMFYUI_BASE,
            "comfyui_username": COMFYUI_USERNAME,
            "sync_folders": SYNC_FOLDERS,
            "platform_name": PLATFORM_NAME,
        }
    )


@app.route("/api/config", methods=["POST"])
def update_config():
    global MODELS_ROOT, S3_BUCKET, S3_PREFIX, S3_ENDPOINT_URL, AWS_PROFILE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, INCLUDE_PERSONAL_STUFF, PERSONAL_PATHS, COMFYUI_BASE, COMFYUI_USERNAME, SYNC_FOLDERS, PLATFORM_NAME
    d = request.json or {}
    if "models_root" in d:
        MODELS_ROOT = os.path.expanduser(d["models_root"])
    if "s3_bucket" in d:
        S3_BUCKET = d["s3_bucket"]
    if "s3_prefix" in d:
        S3_PREFIX = d["s3_prefix"]
    if "s3_endpoint_url" in d:
        S3_ENDPOINT_URL = str(d["s3_endpoint_url"] or "").strip()
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
    if "comfyui_base" in d:
        COMFYUI_BASE = os.path.expanduser(str(d["comfyui_base"] or "/workspace/ComfyUI"))
    if "comfyui_username" in d:
        COMFYUI_USERNAME = str(d["comfyui_username"] or "default").strip() or "default"
    if "sync_folders" in d:
        SYNC_FOLDERS = [str(f).strip() for f in d["sync_folders"] if str(f).strip()]
    if "platform_name" in d:
        PLATFORM_NAME = str(d.get("platform_name") or "").strip()
    try:
        save_settings(
            {
                "models_root": MODELS_ROOT,
                "s3_bucket": S3_BUCKET,
                "s3_prefix": S3_PREFIX,
                "s3_endpoint_url": S3_ENDPOINT_URL,
                "aws_profile": AWS_PROFILE,
                "aws_access_key_id": AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
                "aws_session_token": AWS_SESSION_TOKEN,
                "include_personal_stuff": INCLUDE_PERSONAL_STUFF,
                "personal_paths": PERSONAL_PATHS,
                "comfyui_base": COMFYUI_BASE,
                "comfyui_username": COMFYUI_USERNAME,
                "sync_folders": SYNC_FOLDERS,
                "platform_name": PLATFORM_NAME,
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

    # Refresh S3 key set to avoid unnecessary re-upload when object already exists.
    existing_s3_keys = get_s3_keys_cached(force_refresh=True)
    upload_plan = []
    skipped_pre_count = 0
    total_bytes = 0
    for p in paths:
        s3_key = get_s3_key(p)
        already_on_s3 = s3_key in existing_s3_keys
        upload_plan.append((p, s3_key, already_on_s3))
        if already_on_s3:
            skipped_pre_count += 1
            continue
        if os.path.exists(p):
            total_bytes += os.path.getsize(p)

    jobs[job_id] = {
        "total_files": len(paths),
        "done_files": 0,
        "total_bytes": total_bytes,
        "transferred_bytes": 0,
        "skipped_files": 0,
        "current": "",
        "errors": [],
        "finished": False,
    }

    add_log(
        "info",
        f"Upload started — {len(paths)} file(s), {skipped_pre_count} already on S3",
    )

    def do_upload():
        s3 = get_s3_client()
        job = jobs[job_id]
        for path, s3_key, already_on_s3 in upload_plan:
            job["current"] = os.path.basename(path)
            try:
                if already_on_s3:
                    job["done_files"] += 1
                    job["skipped_files"] += 1
                    add_log("info", f"Skipped (already on S3): {os.path.basename(path)}")
                    continue

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
        skipped = job.get("skipped_files", 0)
        uploaded = job["done_files"] - errs - skipped
        add_log(
            "info",
            f"Upload done — {uploaded} uploaded, {skipped} skipped, {errs} errors",
        )
        job["finished"] = True
        # Avoid stale UI after upload: force next /api/files to rebuild tree with fresh S3 state.
        invalidate_scan_caches()
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
    """Hard-delete local files and/or folders. Only call this after confirming S3 upload succeeded."""
    paths = request.json.get("paths", [])
    print(f"[delete_local] received {len(paths)} path(s): {paths}", flush=True)
    deleted, errors = [], []

    def delete_path(path):
        if os.path.isdir(path):
            import shutil
            shutil.rmtree(path)
            return True
        else:
            os.remove(path)
            return True

    for path in paths:
        try:
            delete_path(path)
            deleted.append(path)
            add_log("success", f"Deleted local: {os.path.basename(path)}")
        except Exception as e:
            errors.append({"path": path, "error": str(e)})
            add_log("error", f"Failed to delete local {os.path.basename(path)}: {e}")
    add_log("info", f"Delete local — {len(deleted)} deleted, {len(errors)} errors")
    with cache_lock:
        files_cache["cache_key"] = None
        files_cache["tree"] = None
        files_cache["scanned_at"] = None
    trigger_background_refresh(force_s3_refresh=False)
    return jsonify(
        {"deleted": len(deleted), "deleted_paths": deleted, "errors": errors}
    )


# --- Sync helpers ---


def _get_sync_s3_prefix() -> str:
    return f"{S3_PREFIX}sync/"


def _list_s3_sync_objects(folders: list) -> dict:
    """Return {rel_path: {key, size, last_modified}} for the given folders under the sync prefix."""
    prefix = _get_sync_s3_prefix()
    result = {}
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel:
                continue
            folder = rel.split("/")[0]
            if folder in folders:
                result[rel] = {
                    "key": key,
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                }
    return result


def _scan_local_sync_files(folders: list) -> dict:
    """Return {rel_path: {path, size, mtime}} for all files in the given sync folders."""
    base = Path(COMFYUI_BASE)
    result = {}
    for folder in folders:
        folder_path = base / folder
        if not folder_path.exists():
            continue
        try:
            for p in sorted(folder_path.rglob("*")):
                if p.is_file():
                    rel = str(p.relative_to(base)).replace(os.sep, "/")
                    stat = p.stat()
                    result[rel] = {"path": str(p), "size": stat.st_size, "mtime": stat.st_mtime}
        except (PermissionError, OSError):
            pass
    return result


# --- Sync endpoints ---


@app.route("/api/sync/status")
def sync_status():
    if not S3_BUCKET:
        return jsonify({"error": "No S3 bucket configured"}), 400

    folders = request.args.getlist("folders") or SYNC_FOLDERS
    base = Path(COMFYUI_BASE)

    try:
        s3_objects = _list_s3_sync_objects(folders)
    except NoCredentialsError:
        return jsonify({"error": "AWS credentials not found"}), 401
    except ProfileNotFound as e:
        return jsonify({"error": f"AWS profile error: {e}"}), 400
    except (ClientError, BotoCoreError, Exception) as e:
        return jsonify({"error": f"S3 error: {e}"}), 500

    local_files = _scan_local_sync_files(folders)

    folder_stats = {}
    for folder in folders:
        local_subset = {k: v for k, v in local_files.items() if k.startswith(f"{folder}/")}
        s3_subset = {k: v for k, v in s3_objects.items() if k.startswith(f"{folder}/")}

        local_keys = set(local_subset)
        s3_keys = set(s3_subset)
        both = local_keys & s3_keys

        folder_stats[folder] = {
            "exists": (base / folder).exists(),
            "local_count": len(local_subset),
            "s3_count": len(s3_subset),
            "local_only_count": len(local_keys - s3_keys),
            "s3_only_count": len(s3_keys - local_keys),
            "in_sync_count": sum(1 for r in both if local_subset[r]["size"] == s3_subset[r]["size"]),
            "different_count": sum(1 for r in both if local_subset[r]["size"] != s3_subset[r]["size"]),
            "local_size": sum(v["size"] for v in local_subset.values()),
            "s3_size": sum(v["size"] for v in s3_subset.values()),
        }

    return jsonify({
        "folders": folder_stats,
        "comfyui_base": COMFYUI_BASE,
        "comfyui_username": COMFYUI_USERNAME,
        "sync_s3_prefix": _get_sync_s3_prefix(),
        "platform_name": PLATFORM_NAME,
    })


@app.route("/api/sync/push", methods=["POST"])
def sync_push():
    data = request.json or {}
    folders = data.get("folders") or SYNC_FOLDERS
    job_id = data.get("job_id", f"sync_push_{int(time.time())}")
    force = bool(data.get("force", False))

    if not S3_BUCKET:
        return jsonify({"error": "No S3 bucket configured"}), 400

    base = Path(COMFYUI_BASE)
    sync_prefix = _get_sync_s3_prefix()

    # Fetch existing S3 objects for smart skip (skip when size matches)
    existing_s3 = {}
    if not force:
        try:
            existing_s3 = {rel: info["size"] for rel, info in _list_s3_sync_objects(folders).items()}
        except Exception as e:
            add_log("warning", f"Sync push: could not fetch existing S3 objects: {e}")

    local_files = _scan_local_sync_files(folders)

    upload_plan = []
    total_bytes = 0
    for rel, info in sorted(local_files.items()):
        s3_key = f"{sync_prefix}{rel}"
        skip = (not force) and rel in existing_s3 and existing_s3[rel] == info["size"]
        upload_plan.append((info["path"], s3_key, info["mtime"], skip))
        if not skip:
            total_bytes += info["size"]

    skipped_pre = sum(1 for _, _, _, s in upload_plan if s)
    jobs[job_id] = {
        "total_files": len(upload_plan),
        "done_files": 0,
        "total_bytes": total_bytes,
        "transferred_bytes": 0,
        "skipped_files": 0,
        "current": "",
        "errors": [],
        "finished": False,
    }

    add_log("info", f"Sync push started — {len(upload_plan)} files, {skipped_pre} already in sync")

    def do_push():
        s3 = get_s3_client()
        job = jobs[job_id]
        for local_path, s3_key, mtime, skip in upload_plan:
            job["current"] = os.path.basename(local_path)
            if skip:
                job["done_files"] += 1
                job["skipped_files"] += 1
                continue
            try:
                def make_cb(j):
                    def cb(n): j["transferred_bytes"] += n
                    return cb
                s3.upload_file(
                    local_path, S3_BUCKET, s3_key,
                    ExtraArgs={"Metadata": {"local-mtime": str(mtime)}},
                    Callback=make_cb(job),
                )
                job["done_files"] += 1
                add_log("success", f"Sync pushed: {os.path.basename(local_path)}")
            except Exception as e:
                job["errors"].append({"path": local_path, "error": str(e)})
                job["done_files"] += 1
                add_log("error", f"Sync push failed {os.path.basename(local_path)}: {e}")
        skipped = job.get("skipped_files", 0)
        errs = len(job["errors"])
        add_log("info", f"Sync push done — {job['done_files'] - errs - skipped} uploaded, {skipped} skipped, {errs} errors")
        job["finished"] = True

    threading.Thread(target=do_push, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/sync/pull", methods=["POST"])
def sync_pull():
    data = request.json or {}
    folders = data.get("folders") or SYNC_FOLDERS
    job_id = data.get("job_id", f"sync_pull_{int(time.time())}")
    force = bool(data.get("force", False))

    if not S3_BUCKET:
        return jsonify({"error": "No S3 bucket configured"}), 400

    base = Path(COMFYUI_BASE)

    try:
        s3_objects = _list_s3_sync_objects(folders)
    except NoCredentialsError:
        return jsonify({"error": "AWS credentials not found"}), 401
    except ProfileNotFound as e:
        return jsonify({"error": f"AWS profile error: {e}"}), 400
    except (ClientError, BotoCoreError, Exception) as e:
        return jsonify({"error": f"S3 error: {e}"}), 500

    download_plan = []
    total_bytes = 0
    for rel, info in sorted(s3_objects.items()):
        local_path = str(base / rel)
        skip = (not force) and os.path.exists(local_path) and os.path.getsize(local_path) == info["size"]
        download_plan.append((info["key"], local_path, info["size"], skip))
        if not skip:
            total_bytes += info["size"]

    skipped_pre = sum(1 for _, _, _, s in download_plan if s)
    jobs[job_id] = {
        "total_files": len(download_plan),
        "done_files": 0,
        "total_bytes": total_bytes,
        "transferred_bytes": 0,
        "skipped_files": 0,
        "current": "",
        "errors": [],
        "finished": False,
    }

    add_log("info", f"Sync pull started — {len(download_plan)} files, {skipped_pre} already in sync")

    def do_pull():
        s3 = get_s3_client()
        job = jobs[job_id]
        for key, local_path, size, skip in download_plan:
            job["current"] = os.path.basename(local_path)
            if skip:
                job["done_files"] += 1
                job["skipped_files"] += 1
                continue
            try:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                def make_cb(j):
                    def cb(n): j["transferred_bytes"] += n
                    return cb
                s3.download_file(S3_BUCKET, key, local_path, Callback=make_cb(job))
                job["done_files"] += 1
                add_log("success", f"Sync pulled: {os.path.basename(local_path)}")
            except Exception as e:
                job["errors"].append({"key": key, "error": str(e)})
                job["done_files"] += 1
                add_log("error", f"Sync pull failed {os.path.basename(local_path)}: {e}")
        skipped = job.get("skipped_files", 0)
        errs = len(job["errors"])
        add_log("info", f"Sync pull done — {job['done_files'] - errs - skipped} downloaded, {skipped} skipped, {errs} errors")
        job["finished"] = True

    threading.Thread(target=do_pull, daemon=True).start()
    return jsonify({"job_id": job_id})


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


@app.route("/api/delete_s3_folder", methods=["POST"])
def delete_s3_folder():
    """Delete a folder (all objects with a given prefix) from S3 bucket."""
    prefix = request.json.get("prefix", "")
    if not prefix:
        return jsonify({"error": "No prefix provided"}), 400
    if not S3_BUCKET:
        return jsonify({"error": "No S3 bucket configured"}), 400

    full_prefix = S3_PREFIX + prefix if not prefix.startswith(S3_PREFIX) else prefix
    print(f"[delete_s3_folder] deleting all objects with prefix: {full_prefix}", flush=True)

    s3 = get_s3_client()
    deleted, errors = 0, []

    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=full_prefix):
            objects = page.get("Contents", [])
            if objects:
                delete_keys = [{"Key": obj["Key"]} for obj in objects]
                s3.delete_objects(
                    Bucket=S3_BUCKET,
                    Delete={"Objects": delete_keys, "Quiet": True}
                )
                deleted += len(delete_keys)
                for obj in objects:
                    add_log("success", f"Deleted from S3: {obj['Key'].split('/')[-1]}")
    except Exception as e:
        errors.append({"prefix": full_prefix, "error": str(e)})
        add_log("error", f"Failed to delete S3 folder {prefix}: {e}")

    add_log("info", f"Delete S3 folder — {deleted} deleted, {len(errors)} errors")
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


def _restart_process():
    # Give Flask time to send HTTP response before replacing process.
    time.sleep(0.3)
    try:
        script_path = os.path.abspath(__file__)
        restart_cmd = [sys.executable, script_path, "--port", str(args.port)]
        add_log("warning", f"Restart (kill + relaunch): {' '.join(restart_cmd)}")

        # Launch next instance with a slight delay so current process can terminate first
        launch_cmd = (
            f"sleep 0.6; "
            f"exec {sys.executable} {script_path} --port {args.port}"
        )
        subprocess.Popen(
            ["/bin/bash", "-lc", launch_cmd],
            cwd=os.path.dirname(script_path) or ".",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Kill current process after scheduling relaunch
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception as e:
        add_log("error", f"Restart failed: {e}")


@app.route("/api/restart", methods=["POST"])
def restart_app():
    add_log("warning", "Application restart requested from UI")
    threading.Thread(target=_restart_process, daemon=True).start()
    return jsonify({"status": "restarting"})


if __name__ == "__main__":
    print(f"🚀 S3 Offloader → http://localhost:{args.port}")
    print(f"⚙️ Settings file: {CONFIG_FILE}")
    print(f"📁 Models root : {MODELS_ROOT}")
    print(f"🪣 S3 bucket   : {S3_BUCKET or '(not set)'}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
