#!/usr/bin/env python3
"""
Download latest app scripts from S3/R2 at pod startup.
Called by start_wrapper.sh before launching app.py.
Reads credentials from environment variables (with settings.json as fallback).
Usage: python3 self_update.py
"""
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent
SETTINGS_FILE = REPO_DIR / "settings.json"
SCRIPT_FILES = ["app.py", "start_wrapper.sh", "self_update.py"]


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_config() -> dict:
    s = load_settings()
    return {
        "bucket": os.environ.get("S3_BUCKET") or s.get("s3_bucket", ""),
        "access_key": os.environ.get("AWS_ACCESS_KEY_ID") or s.get("aws_access_key_id", ""),
        "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY") or s.get("aws_secret_access_key", ""),
        "endpoint": (
            os.environ.get("S3_ENDPOINT_URL")
            or os.environ.get("R2_URL")
            or s.get("s3_endpoint_url", "")
            or s.get("r2_url", "")
        ),
        "prefix": os.environ.get("S3_PREFIX") or s.get("s3_prefix", "models-offload/"),
    }


def main() -> int:
    cfg = get_config()
    bucket = str(cfg["bucket"] or "").strip()
    access_key = str(cfg["access_key"] or "").strip()
    secret_key = str(cfg["secret_key"] or "").strip()
    endpoint = str(cfg["endpoint"] or "").strip()
    prefix = str(cfg["prefix"] or "models-offload/").strip()
    scripts_prefix = prefix.rstrip("/") + "/scripts/"

    if not bucket or not access_key or not secret_key:
        print("[self_update] S3 not configured — skipping script update", flush=True)
        return 0

    try:
        import boto3
    except ImportError:
        print("[self_update] boto3 not available — skipping", flush=True)
        return 0

    try:
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        client_kwargs: dict = {}
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint
            client_kwargs["config"] = boto3.session.Config(s3={"addressing_style": "path"})
        s3 = session.client("s3", **client_kwargs)

        resp = s3.list_objects_v2(Bucket=bucket, Prefix=scripts_prefix)
        objects: dict = {}
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            name = key[len(scripts_prefix):]
            if name and "/" not in name and name in SCRIPT_FILES:
                objects[name] = obj

        if not objects:
            print("[self_update] No scripts in S3 — skipping", flush=True)
            return 0

        updated = []
        for filename in SCRIPT_FILES:
            if filename not in objects:
                continue
            obj = objects[filename]
            local_path = REPO_DIR / filename
            s3_size = obj["Size"]
            local_size = local_path.stat().st_size if local_path.exists() else -1
            if local_size == s3_size:
                print(f"[self_update] {filename}: up to date ({s3_size} B)", flush=True)
                continue
            print(f"[self_update] Downloading {filename} ({s3_size} B)...", flush=True)
            tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
            s3.download_file(bucket, obj["Key"], str(tmp_path))
            tmp_path.replace(local_path)
            if filename.endswith(".sh"):
                local_path.chmod(local_path.stat().st_mode | 0o111)
            updated.append(filename)
            print(f"[self_update] {filename}: updated ✓", flush=True)

        if updated:
            print(f"[self_update] Scripts updated: {', '.join(updated)}", flush=True)
        return 0

    except Exception as e:
        print(f"[self_update] Error: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
