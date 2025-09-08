import os, json, tempfile, shutil, subprocess, uuid
import boto3, requests
import runpod


AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=AWS_REGION)

def _download(url, dst):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

def _upload(path, bucket, key):
    s3.upload_file(path, bucket, key)

def _presign(bucket, key, ttl=3600):
    return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=ttl)

def handler(event):
    """
    Input:
    {
      "video_urls": ["https://presigned-clip1.mp4", "..."],
      "s3_bucket": "optional-override",
      "s3_prefix": "facebook-reelmaker/processed/<video_id>/",
      "generate_presigned": true
    }
    """
    data = event.get("input") or {}
    video_urls = data.get("video_urls") or []
    if not video_urls:
        single = data.get("video_url")
        if single:
            video_urls = [single]
    if not video_urls:
        return {"error": "Provide video_urls (list) or video_url (single)."}

    out_bucket = data.get("s3_bucket") or AWS_S3_BUCKET
    out_prefix = data.get("s3_prefix") or f"facebook-reelmaker/outputs/{uuid.uuid4().hex}/"
    gen_presigned = bool(data.get("generate_presigned", True))

    if not out_bucket:
        return {"error": "No S3 bucket configured. Set AWS_S3_BUCKET env or pass s3_bucket in input."}

    work_root = tempfile.mkdtemp(prefix="reelmaker_")
    in_dir = os.path.join(work_root, "in")
    os.makedirs(in_dir, exist_ok=True)

    try:
        # download inputs
        for i, url in enumerate(video_urls):
            name = os.path.basename(url.split("?")[0]) or f"input_{i}.mp4"
            if not name.lower().endswith(".mp4"):
                name = f"{name}.mp4"
            _download(url, os.path.join(in_dir, name))

        # run your existing script unchanged
        # (reelmaker writes to <in_dir>/processed_videos)
        proc = subprocess.run(
            ["python", "reelmaker.py", in_dir],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        stdout_tail = proc.stdout[-2000:] if proc.stdout else ""
        if proc.returncode != 0:
            return {"error": "reelmaker.py failed", "stdout": stdout_tail}

        out_dir = os.path.join(in_dir, "processed_videos")
        if not os.path.isdir(out_dir):
            return {"error": f"No output dir at {out_dir}", "stdout": stdout_tail}

        files = []
        for fname in os.listdir(out_dir):
            if not fname.lower().endswith(".mp4"):
                continue
            fpath = os.path.join(out_dir, fname)
            key = os.path.join(out_prefix, fname).replace("\\", "/")
            _upload(fpath, out_bucket, key)
            item = {"name": fname, "bucket": out_bucket, "s3_key": key}
            if gen_presigned:
                item["url"] = _presign(out_bucket, key)
            files.append(item)

        return {"files": files, "stdout": stdout_tail}
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

runpod.serverless.start({"handler": handler})
