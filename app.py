"""
S3 Model Offloader — Flask backend
Scans local model directories, uploads to S3 with path metadata for 1-click restore.
"""

import os
import json
import threading
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

# --- Config ---
MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".ggml", ".pkl", ".q4_0", ".q8_0"}
MODELS_ROOT = os.path.expanduser(os.getenv("MODELS_ROOT", "~/models"))
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "models-offload/")
AWS_PROFILE = os.getenv("AWS_PROFILE", None)

# In-memory progress tracking
upload_progress = {}
download_progress = {}

def get_s3_client():
    session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else boto3.Session()
    return session.client("s3")

def get_s3_resource():
    session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else boto3.Session()
    return session.resource("s3")

def is_model_file(path: Path) -> bool:
    return path.suffix.lower() in MODEL_EXTENSIONS

def get_s3_key(local_path: str) -> str:
    """Build S3 key: prefix + relative path from MODELS_ROOT"""
    rel = os.path.relpath(local_path, MODELS_ROOT)
    return S3_PREFIX + rel.replace("\\", "/")

def format_size(bytes_size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


# --- API Routes ---

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/config")
def get_config():
    return jsonify({
        "models_root": MODELS_ROOT,
        "s3_bucket": S3_BUCKET,
        "s3_prefix": S3_PREFIX,
    })

@app.route("/api/config", methods=["POST"])
def update_config():
    global MODELS_ROOT, S3_BUCKET, S3_PREFIX, AWS_PROFILE
    data = request.json
    if "models_root" in data:
        MODELS_ROOT = os.path.expanduser(data["models_root"])
    if "s3_bucket" in data:
        S3_BUCKET = data["s3_bucket"]
    if "s3_prefix" in data:
        S3_PREFIX = data["s3_prefix"]
    if "aws_profile" in data:
        AWS_PROFILE = data["aws_profile"] or None
    return jsonify({"status": "ok"})

@app.route("/api/files")
def list_files():
    """Return directory tree of model files with S3 status"""
    root = Path(MODELS_ROOT)
    if not root.exists():
        return jsonify({"error": f"Directory not found: {MODELS_ROOT}"}), 404

    # Check which files exist on S3
    s3_keys = set()
    if S3_BUCKET:
        try:
            s3 = get_s3_client()
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
                for obj in page.get("Contents", []):
                    s3_keys.add(obj["Key"])
        except Exception:
            pass  # Can't reach S3, just show local files

    def build_tree(path: Path, rel_root: Path):
        node = {
            "name": path.name,
            "path": str(path),
            "rel_path": str(path.relative_to(rel_root)),
            "type": "dir" if path.is_dir() else "file",
        }
        if path.is_dir():
            children = []
            try:
                for child in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                    if child.is_dir() or is_model_file(child):
                        children.append(build_tree(child, rel_root))
            except PermissionError:
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

    tree = build_tree(root, root)
    return jsonify(tree)

@app.route("/api/s3/list")
def list_s3():
    """List files that are on S3 (for restore view)"""
    if not S3_BUCKET:
        return jsonify({"error": "No S3 bucket configured"}), 400
    try:
        s3 = get_s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        files = []
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(S3_PREFIX):]
                local_path = os.path.join(MODELS_ROOT, rel)
                files.append({
                    "s3_key": key,
                    "rel_path": rel,
                    "local_path": local_path,
                    "size": obj["Size"],
                    "size_human": format_size(obj["Size"]),
                    "local_exists": os.path.exists(local_path),
                    "last_modified": obj["LastModified"].isoformat(),
                })
        return jsonify(files)
    except NoCredentialsError:
        return jsonify({"error": "AWS credentials not found"}), 401
    except ClientError as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
def upload_files():
    """Upload selected files to S3"""
    data = request.json
    paths = data.get("paths", [])
    job_id = data.get("job_id", "upload_job")

    if not S3_BUCKET:
        return jsonify({"error": "No S3 bucket configured"}), 400

    upload_progress[job_id] = {"total": len(paths), "done": 0, "errors": [], "current": ""}

    def do_upload():
        s3 = get_s3_client()
        for i, path in enumerate(paths):
            upload_progress[job_id]["current"] = os.path.basename(path)
            try:
                s3_key = get_s3_key(path)
                file_size = os.path.getsize(path)

                # Track per-file progress via callback
                uploaded = [0]
                def callback(bytes_transferred):
                    uploaded[0] += bytes_transferred
                    pct = int(uploaded[0] / file_size * 100) if file_size > 0 else 100
                    upload_progress[job_id]["file_pct"] = pct

                s3.upload_file(path, S3_BUCKET, s3_key, Callback=callback)
                upload_progress[job_id]["done"] += 1
            except Exception as e:
                upload_progress[job_id]["errors"].append({"path": path, "error": str(e)})
                upload_progress[job_id]["done"] += 1

        upload_progress[job_id]["finished"] = True

    thread = threading.Thread(target=do_upload, daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})

@app.route("/api/upload/progress/<job_id>")
def upload_progress_check(job_id):
    return jsonify(upload_progress.get(job_id, {"error": "Job not found"}))

@app.route("/api/restore", methods=["POST"])
def restore_files():
    """Download files from S3 back to their original local path"""
    data = request.json
    keys = data.get("keys", [])  # list of s3_key strings
    job_id = data.get("job_id", "restore_job")

    if not S3_BUCKET:
        return jsonify({"error": "No S3 bucket configured"}), 400

    download_progress[job_id] = {"total": len(keys), "done": 0, "errors": [], "current": ""}

    def do_restore():
        s3 = get_s3_resource()
        for key in keys:
            rel = key[len(S3_PREFIX):]
            local_path = os.path.join(MODELS_ROOT, rel)
            download_progress[job_id]["current"] = os.path.basename(local_path)
            try:
                # Re-create directory structure
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                obj = s3.Object(S3_BUCKET, key)
                file_size = obj.content_length

                downloaded = [0]
                def callback(bytes_transferred):
                    downloaded[0] += bytes_transferred
                    pct = int(downloaded[0] / file_size * 100) if file_size > 0 else 100
                    download_progress[job_id]["file_pct"] = pct

                s3.Bucket(S3_BUCKET).download_file(key, local_path, Callback=callback)
                download_progress[job_id]["done"] += 1
            except Exception as e:
                download_progress[job_id]["errors"].append({"key": key, "error": str(e)})
                download_progress[job_id]["done"] += 1

        download_progress[job_id]["finished"] = True

    thread = threading.Thread(target=do_restore, daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})

@app.route("/api/restore/progress/<job_id>")
def restore_progress_check(job_id):
    return jsonify(download_progress.get(job_id, {"error": "Job not found"}))

@app.route("/api/delete_s3", methods=["POST"])
def delete_from_s3():
    """Remove files from S3 (after confirming restore)"""
    data = request.json
    keys = data.get("keys", [])
    if not S3_BUCKET:
        return jsonify({"error": "No S3 bucket configured"}), 400
    try:
        s3 = get_s3_client()
        for key in keys:
            s3.delete_object(Bucket=S3_BUCKET, Key=key)
        return jsonify({"deleted": len(keys)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"🚀 S3 Offloader running → http://localhost:5050")
    print(f"📁 Models root: {MODELS_ROOT}")
    print(f"🪣 S3 bucket: {S3_BUCKET or '(not set — configure in UI)'}")
    app.run(host="0.0.0.0", port=5050, debug=False)
