import os, json, subprocess, tempfile, logging
import boto3
from botocore.client import Config
import requests
import runpod

LOG_LEVEL = os.getenv("LOG_LEVEL","INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("reelmaker")

AWS_REGION     = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET  = os.getenv("AWS_S3_BUCKET")
S3_PREFIX_BASE = os.getenv("S3_PREFIX_BASE", "jobs")

if not AWS_S3_BUCKET:
    raise RuntimeError("AWS_S3_BUCKET must be set in the environment for S3-only operation.")

s3 = boto3.client("s3", region_name=AWS_REGION, config=Config(s3={"addressing_style":"virtual"}))

def _merge_event(event):
    d = {}
    if isinstance(event, dict):
        inp = event.get("input")
        if isinstance(inp, dict):
            d.update(inp)
        d.update({k:v for k,v in event.items() if k != "input"})
    return d


def _key(job_id: str, *parts: str) -> str:
    safe = [p.strip("/").replace("\\","/") for p in parts if p]
    return "/".join([S3_PREFIX_BASE.strip("/"), job_id] + safe)

def _presign(key: str, expires=604800) -> str:
    return s3.generate_presigned_url("get_object", Params={"Bucket": AWS_S3_BUCKET, "Key": key}, ExpiresIn=expires)

def _download(url: str, dest: str):
    with requests.get(url, stream=True, timeout=1800) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)

def _ffmpeg_reel(in_path: str, out_path: str):
    # Portrait 1080x1920: scale to fit, pad to 1080x1920
    vf = "scale=1080:-2:force_original_aspect_ratio=decrease,pad=1080:1920:(1080-iw)/2:(1920-ih)/2"
    cmd = [
        "ffmpeg","-hide_banner","-y","-i", in_path,
        "-vf", vf,
        "-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","160k",
        "-movflags","+faststart",
        out_path
    ]
    subprocess.check_call(cmd)


def _reelmaker_process(in_path: str, out_path: str):
    """Run reelmaker.py on the temp directory containing in.mp4 and move the first processed_* output to out_path."""
    import pathlib, glob, shutil, sys, subprocess, os
    d = os.path.dirname(in_path) or "."
    # Execute reelmaker.py against the directory
    subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "reelmaker.py"), d])
    # Collect output from processed_videos/processed_*.mp4
    proc_dir = os.path.join(d, "processed_videos")
    candidates = sorted(glob.glob(os.path.join(proc_dir, "processed_*.mp4")))
    if not candidates:
        raise RuntimeError("reelmaker produced no output")
    shutil.move(candidates[0], out_path)

def handler(event):
    """Input:
       {
         "job_id": "20250920-...-abc123",
         "video_url": "https://presigned-or-public-url.mp4",
         "output_basename": "reel.mp4"   # optional
       }
    """
    log.info(f"event: {json.dumps(event)[:500]}")
    data = _merge_event(event)
    job_id = data.get("job_id") or os.getenv("JOB_ID")
    in_url = data.get("video_url") or data.get("input_url") or data.get("source_url") or data.get("video")
    out_name = data.get("output_basename", "reel.mp4")

    if not job_id:
        return {"ok": False, "error": "job_id is required"}
    if not in_url or not str(in_url).startswith("http"):
        return {"ok": False, "error": "video_url (http/https) is required"}

    tdir = tempfile.mkdtemp(prefix="reel_")
    in_path  = os.path.join(tdir, "in.mp4")
    out_path = os.path.join(tdir, out_name)

    # 1) Download input
    log.info("Downloading input video...")
    _download(in_url, in_path)

    # 2) ffmpeg → portrait
    log.info("Running reelmaker face-follow crop (1080x1920)...")
    _reelmaker_process(in_path, out_path)

    # 3) Upload to S3
    key = _key(job_id, "reels", out_name)
    log.info(f"Uploading to s3://{AWS_S3_BUCKET}/{key}")
    s3.upload_file(out_path, AWS_S3_BUCKET, key)

    # 4) Return keys + presigned URL
    url = _presign(key)
    return {
        "ok": True,
        "keys": {"reel_key": key},
        "s3":   {"reel": f"s3://{AWS_S3_BUCKET}/{key}"},
        "urls": {"reel_url": url},
    }

runpod.serverless.start({"handler": handler})
