import os, io, sys, json, tempfile, traceback, subprocess, shutil, time, boto3
from urllib.parse import urlparse
import requests

# wire up Slack
SLACK_WEBHOOK_ENV = os.environ.get("SLACK_WEBHOOK", "").strip()
SLACK_VERBOSE = os.environ.get("SLACK_VERBOSE", "false").lower() in ("1","true","yes","on")
POD_NAME = os.environ.get("POD_NAME", "facebook-reelmaker-runpod")

def _slack_is_important(message: str) -> bool:
    msg = (message or "").lower()
    return any(tok in msg for tok in (":x:", ":warning:", "[error]", "error:", "failed"))

def post_to_slack(msg: str, webhook_override: str | None = None):
    if not (SLACK_VERBOSE or _slack_is_important(msg)):
        return
    url = (webhook_override or SLACK_WEBHOOK_ENV or "").strip()
    if not url:
        return
    try:
        out = f"[{POD_NAME}] {msg}"
        requests.post(url, json={"text": out}, timeout=5)
    except Exception:
        pass

def human_bytes(n: int | None) -> str:
    if not n:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"

def http_get_to(path: str, url: str, slack=None):
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        size = int(r.headers.get("Content-Length") or 0)
        ctype = r.headers.get("Content-Type") or "?"
        print(f"[FETCH] {url} -> {path} ({human_bytes(size)}; {ctype})", flush=True)
        if slack:
            post_to_slack(f"[reelmaker] downloading input… size={human_bytes(size)} type={ctype}", slack)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

def sh(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return (0, out.strip())
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.output.strip())

def ffprobe_summary(path: str) -> str:
    code, out = sh([
        "ffprobe", "-hide_banner", "-v", "error",
        "-show_entries", "format=duration:stream=index,codec_type,codec_name,avg_frame_rate,width,height",
        "-of", "json", path
    ])
    if code == 0:
        return out
    return f"(ffprobe failed)\n{out}"

def ffmpeg_null_mux(path: str) -> str:
    # Surface decode/packetization errors without writing output
    code, out = sh(["ffmpeg", "-hide_banner", "-v", "error", "-i", path, "-f", "null", "-"])
    if code == 0 and not out:
        return "(ffmpeg null mux OK; no errors)"
    return out or "(no output from ffmpeg)"

def list_tree(root: str) -> str:
    buf = io.StringIO()
    for base, dirs, files in os.walk(root):
        rel = os.path.relpath(base, root)
        buf.write(f"./{rel}\n")
        for name in files:
            p = os.path.join(base, name)
            try:
                sz = os.path.getsize(p)
            except Exception:
                sz = -1
            buf.write(f"  {name}  ({human_bytes(sz)})\n")
    return buf.getvalue()

# -------- S3 helpers --------
def s3_client(region: str | None = None):
    region = region or os.environ.get("AWS_REGION") or "us-east-1"
    return boto3.client("s3", region_name=region)

def s3_upload_file(bucket: str, key: str, local_path: str, region: str | None = None):
    c = s3_client(region)
    c.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"

def s3_presign_get(bucket: str, key: str, region: str | None = None, expires: int = 604800):
    c = s3_client(region)
    return c.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)

def s3_head(bucket: str, key: str, region: str | None = None) -> dict | None:
    try:
        c = s3_client(region)
        return c.head_object(Bucket=bucket, Key=key)
    except Exception:
        return None

def s3_download_to(bucket: str, key: str, local_path: str, region: str | None = None, slack=None):
    c = s3_client(region)
    meta = c.head_object(Bucket=bucket, Key=key)
    size = int(meta.get("ContentLength", 0))
    ctype = (meta.get("ContentType") or "?")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    c.download_file(bucket, key, local_path)
    print(f"[S3 FETCH] s3://{bucket}/{key} -> {local_path} ({human_bytes(size)}; {ctype})", flush=True)
    if slack:
        post_to_slack(f"[reelmaker] downloading input from S3… size={human_bytes(size)} type={ctype}", slack)
    return size, ctype

# ---- import reelmaker core ----
import reelmaker as rm

def _reelmaker_process(in_path: str, out_dir: str, slack=None):
    # reelmaker.py expects a folder of mp4s and writes to processed_videos/
    work_dir = tempfile.mkdtemp(prefix="reel_")
    inp_dir = work_dir
    proc_dir = os.path.join(work_dir, "processed_videos")
    os.makedirs(proc_dir, exist_ok=True)
    # place the single input file into the folder
    base = os.path.basename(in_path)
    local_in = os.path.join(inp_dir, base)
    shutil.copyfile(in_path, local_in)

    print(f"[reelmaker] analyze+process start… work={work_dir}", flush=True)
    post_to_slack(f"[reelmaker] face-follow crop (1080x1920)…", slack)

    try:
        # Call main pipeline on the folder
        face_pos, motion_pos, fps, frame_dims = rm.analyze_video(local_in)
        crop_rects = rm.compute_crop_positions(face_pos, motion_pos, frame_dims, fps)

        import moviepy.editor as mpy
        clip = mpy.VideoFileClip(local_in)

        def process_frame(get_frame, t):
            frame_idx = int(t * fps)
            if frame_idx >= len(crop_rects):
                frame_idx = len(crop_rects) - 1
            x1, y1, w, h = crop_rects[frame_idx]
            frame = get_frame(t)
            import cv2
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cropped = bgr[y1:y1+h, x1:x1+w]
            resized = cv2.resize(cropped, (rm.DESIRED_WIDTH, rm.DESIRED_HEIGHT), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        processed = clip.fl(process_frame)

        out_file = os.path.join(proc_dir, f"processed_{base}")
        processed.write_videofile(
            out_file,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(work_dir, "temp-audio.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None
        )
    except Exception:
        # bubble up; outer handler will Slack the diagnostics
        raise
    finally:
        # always show what we ended up with
        listing = list_tree(work_dir)
        print(f"[reelmaker] work dir listing:\n{listing}", flush=True)

    # Return the output (if any)
    outs = [f for f in os.listdir(proc_dir) if f.lower().endswith(".mp4")]
    post_to_slack(f"[reelmaker] found outputs: {outs}", slack)
    return work_dir, proc_dir, sorted(outs)

def _valid_media_ffprobe_json(ffprobe_json: str) -> bool:
    try:
        obj = json.loads(ffprobe_json or "{}")
        fmt = obj.get("format") or {}
        streams = obj.get("streams") or []
        # duration must be present and parsable; at least one stream
        dur = float(fmt.get("duration", "nan"))
        return (not (dur != dur)) and (dur > 0) and (len(streams) > 0)  # NaN check
    except Exception:
        return False

def handler(event):
    inp = (event or {}).get("input") or {}
    job_id = inp.get("job_id") or "noid"

    # Preferred S3 path
    bucket = (inp.get("bucket") or (inp.get("s3") or {}).get("bucket") or os.environ.get("AWS_S3_BUCKET") or "").strip()
    video_key = (inp.get("video_key") or (inp.get("s3") or {}).get("video_key") or "").strip()

    # Fallback URL path
    video_url = inp.get("video_url") or inp.get("mp4_url") or inp.get("url")

    output_basename = inp.get("output_basename") or "output.mp4"
    slack = inp.get("slack_webhook") or None

    # S3 output config (keep your previous semantics)
    s3_cfg = inp.get("s3") or {}
    s3_bucket = (s3_cfg.get("bucket") or os.environ.get("AWS_S3_BUCKET") or "").strip()
    s3_region = (s3_cfg.get("region") or os.environ.get("AWS_REGION") or "us-east-1").strip()
    s3_key = (s3_cfg.get("key") or f"jobs/{job_id}/reels/{output_basename}").strip()

    with tempfile.TemporaryDirectory(prefix="reel_dl_") as td:
        in_path = os.path.join(td, "input.mp4")

        # --- Download preference: S3 key -> URL
        size_bytes = 0
        try:
            if bucket and video_key:
                head = s3_head(bucket, video_key, s3_region)
                if not head:
                    msg = f"cannot head_object s3://{bucket}/{video_key}"
                    post_to_slack(f":x: [reelmaker] {job_id} {msg}", slack)
                    return {"error": "s3_head_failed", "details": msg}
                size_bytes = int(head.get("ContentLength", 0))
                if size_bytes < 50 * 1024:
                    msg = f"S3 object too small ({size_bytes} B) s3://{bucket}/{video_key}"
                    post_to_slack(f":x: [reelmaker] {job_id} {msg}", slack)
                    return {"error": "tiny_source", "details": msg}

                s3_download_to(bucket, video_key, in_path, s3_region, slack=slack)

            elif video_url and str(video_url).startswith("http"):
                # optional: quick HEAD to guard 261 B bogus files
                try:
                    r = requests.head(video_url, timeout=20, allow_redirects=True)
                    clen = int(r.headers.get("Content-Length") or 0)
                    if clen and clen < 50 * 1024:
                        msg = f"remote object too small via URL ({clen} B)"
                        post_to_slack(f":x: [reelmaker] {job_id} {msg}", slack)
                        return {"error": "tiny_source_url", "details": msg}
                except Exception:
                    # ignore HEAD errors; GET will raise if truly bad
                    pass

                http_get_to(in_path, video_url, slack)
                size_bytes = os.path.getsize(in_path)
                if size_bytes < 50 * 1024:
                    msg = f"downloaded file too small ({size_bytes} B) from URL"
                    post_to_slack(f":x: [reelmaker] {job_id} {msg}", slack)
                    return {"error": "tiny_download", "details": msg}
            else:
                msg = f"missing input (need bucket+video_key or video_url)"
                post_to_slack(f":x: [reelmaker] {job_id} {msg}", slack)
                return {"error": "missing_input", "details": msg}
        except Exception as e:
            post_to_slack(f":x: [reelmaker] {job_id} download failed: {e}", slack)
            return {"error": "download_failed", "details": str(e)}

        # --- preflight diagnostics
        probe = ffprobe_summary(in_path)
        nullmux = ffmpeg_null_mux(in_path)
        print(f"[diag] ffprobe:\n{probe}\n", flush=True)
        print(f"[diag] ffmpeg null mux:\n{nullmux}\n", flush=True)

        # fail fast if ffprobe says no streams/duration
        if not _valid_media_ffprobe_json(probe):
            post_to_slack(
                f":x: [reelmaker] {job_id} invalid media after download "
                f"({human_bytes(size_bytes)}). See ffprobe/null below.",
                slack
            )
            return {"error": "invalid_media", "details": "ffprobe shows no streams/duration", "probe": probe[:1000]}

        try:
            work_dir, proc_dir, outs = _reelmaker_process(in_path, td, slack)
        except Exception as e:
            tb = traceback.format_exc(limit=8)
            listing = list_tree(work_dir) if 'work_dir' in locals() else "(no work_dir)"
            post_to_slack(
                f":x: [reelmaker] {job_id} processing error\n"
                f"input: {os.path.basename(in_path)}\n"
                f"ffprobe: ```{probe[:3500]}```\n"
                f"ffmpeg null: ```{nullmux[:3500]}```\n"
                f"traceback: ```{tb[:3500]}```\n"
                f"work tree:\n```{listing[:3500]}```",
                slack
            )
            return {"error": "processing_failed", "details": str(e)}

        if not outs:
            listing = list_tree(work_dir)
            post_to_slack(
                f":x: [reelmaker] {job_id} produced no output.\n"
                f"ffprobe: ```{probe[:3500]}```\n"
                f"ffmpeg null: ```{nullmux[:3500]}```\n"
                f"work tree:\n```{listing[:3500]}```",
                slack
            )
            raise RuntimeError(f"reelmaker produced no output in {proc_dir}. Contents: []")

        # success path: put file where the caller expects it (return name + temp path info)
        produced = os.path.join(proc_dir, outs[0])
        final_out = os.path.join(td, output_basename)
        shutil.copyfile(produced, final_out)

        # Upload to S3 if configured
        s3_uri = None
        s3_url = None
        if s3_bucket:
            try:
                s3_uri = s3_upload_file(s3_bucket, s3_key, final_out, s3_region)
                s3_url = s3_presign_get(s3_bucket, s3_key, s3_region)
                post_to_slack(f"[reelmaker] {job_id} uploaded → {s3_uri}", slack)
            except Exception as e:
                post_to_slack(f":x: [reelmaker] {job_id} S3 upload failed: {e}", slack)

        post_to_slack(f"[reelmaker] {job_id} OK → {output_basename}", slack)
        return {
            "ok": True,
            "job_id": job_id,
            "output_path": final_out,
            "produced": outs,
            "probe": probe[:1000],
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "s3_uri": s3_uri,
            "s3_url": s3_url
        }


# runpod glue
try:
    import runpod
    runpod.serverless.start({"handler": handler})
except Exception:
    # allow local run for debugging without runpod
    if __name__ == "__main__":
        print("Runpod not available; this file is meant to be used in serverless mode.")
