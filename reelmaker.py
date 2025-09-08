import cv2
import numpy as np
import moviepy.editor as mpy
import sys
import os

# --- Change 1: Dynamic BASE_DIR to the script's directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Change 2: Use BASE_DIR for model and config file ---
MODEL_FILE = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
CONFIG_FILE = os.path.join(BASE_DIR, "deploy.prototxt")

# Desired output video dimensions (for 9:16 aspect ratio)
DESIRED_WIDTH = 1080
DESIRED_HEIGHT = 1920

TOP_MARGIN_PERCENT = 0.2
BOTTOM_MARGIN_PERCENT = 0.2
LEFT_MARGIN_PERCENT = 0.05
RIGHT_MARGIN_PERCENT = 0.05

INITIAL_PERIOD_SECONDS = 5.0
FRAMES_PER_SEGMENT_SHORT = 15
FRAMES_PER_SEGMENT_LONG = 30
MAX_SPEED = 3
MIN_MOVEMENT_THRESHOLD = 100

DEBUG = False

def load_face_detector():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)
    return net

def detect_faces_dnn(net, frame):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        [104.0, 177.0, 123.0],
        False,
        False,
    )
    net.setInput(blob)
    detections = net.forward()

    h, w = frame.shape[:2]
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faces.append([x1, y1, x2, y2])
    return faces

def compute_motion_between_frames(prev_gray, curr_gray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return thresh

def analyze_video(input_video):
    net = load_face_detector()
    clip = mpy.VideoFileClip(input_video)
    fps = clip.fps
    frame_dims = (int(clip.size[1]), int(clip.size[0]))  # (height, width)

    face_positions = []
    motion_positions = []
    prev_gray = None
    
    for frame in clip.iter_frames():
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = detect_faces_dnn(net, frame_bgr)
        if faces:
            largest_face = max(faces, key=lambda rect: (rect[2]-rect[0]) * (rect[3]-rect[1]))
            face_positions.append(largest_face)
        else:
            face_positions.append(None)
        
        if prev_gray is not None:
            motion_mask = compute_motion_between_frames(prev_gray, curr_gray)
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                motion_positions.append([x, y, w, h])
            else:
                motion_positions.append(None)
        else:
            motion_positions.append(None)

        prev_gray = curr_gray
    
    return face_positions, motion_positions, fps, frame_dims

def compute_crop_positions(face_positions, motion_positions, frame_dims, fps, output_aspect_ratio=9/16.0):
    h_frame, w_frame = frame_dims

    top_limit = int(h_frame * TOP_MARGIN_PERCENT)
    bottom_limit = int(h_frame * (1 - BOTTOM_MARGIN_PERCENT))
    left_limit = int(w_frame * LEFT_MARGIN_PERCENT)
    right_limit = int(w_frame * (1 - RIGHT_MARGIN_PERCENT))

    centers = []
    for face_bbox, motion_bbox in zip(face_positions, motion_positions):
        candidate_center = None
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            cx = x1 + (x2 - x1) // 2
            cy = y1 + (y2 - y1) // 2
            candidate_center = (cx, cy)
        elif motion_bbox is not None:
            x_motion, y_motion, w_motion, h_motion = motion_bbox
            cx = x_motion + w_motion // 2
            cy = y_motion + h_motion // 2
            candidate_center = (cx, cy)
        
        if candidate_center is not None:
            cx, cy = candidate_center
            if cx < left_limit or cx > right_limit or cy < top_limit or cy > bottom_limit:
                centers.append(None)
            else:
                centers.append((cx, cy))
        else:
            centers.append(None)

    for i in range(len(centers)):
        if centers[i] is None:
            if i > 0:
                centers[i] = centers[i - 1]
            else:
                centers[i] = (w_frame // 2, h_frame // 2)

    total_frames = len(centers)
    initial_period_frames = int(INITIAL_PERIOD_SECONDS * fps)

    segment_boundaries = []
    index = 0
    while index < min(initial_period_frames, total_frames):
        end_idx = min(index + FRAMES_PER_SEGMENT_SHORT, initial_period_frames, total_frames)
        segment_boundaries.append((index, end_idx))
        index = end_idx
    while index < total_frames:
        end_idx = min(index + FRAMES_PER_SEGMENT_LONG, total_frames)
        segment_boundaries.append((index, end_idx))
        index = end_idx

    segment_centers = []
    for start_idx, end_idx in segment_boundaries:
        segment = centers[start_idx:end_idx]
        avg_x = int(np.mean([c[0] for c in segment]))
        avg_y = int(np.mean([c[1] for c in segment]))
        segment_centers.append((avg_x, avg_y))

    output_width = int(h_frame * output_aspect_ratio)
    output_height = h_frame
    if output_width > w_frame:
        output_width = w_frame
        output_height = int(w_frame / output_aspect_ratio)

    crop_rects = []
    if not segment_centers:
        for i in range(total_frames):
            x1_crop = (w_frame - output_width) // 2
            y1_crop = (h_frame - output_height) // 2
            crop_rects.append((x1_crop, y1_crop, output_width, output_height))
        return crop_rects

    current_x, current_y = segment_centers[0]
    for frame_idx in range(total_frames):
        seg_idx = None
        for s_idx, (start_idx, end_idx) in enumerate(segment_boundaries):
            if start_idx <= frame_idx < end_idx:
                seg_idx = s_idx
                break

        if seg_idx is None:
            target_x, target_y = segment_centers[-1]
        else:
            current_seg_start, current_seg_end = segment_boundaries[seg_idx]
            seg_duration = current_seg_end - current_seg_start
            if seg_duration <= 1:
                target_x, target_y = segment_centers[seg_idx]
            else:
                progress = (frame_idx - current_seg_start) / float(seg_duration)
                current_center = segment_centers[seg_idx]
                if seg_idx + 1 < len(segment_centers):
                    next_center = segment_centers[seg_idx + 1]
                else:
                    next_center = current_center
                interp_x = int((1 - progress) * current_center[0] + progress * next_center[0])
                interp_y = int((1 - progress) * current_center[1] + progress * next_center[1])
                target_x, target_y = interp_x, interp_y

        dx = target_x - current_x
        dy = target_y - current_y
        dist = np.sqrt(dx*dx + dy*dy)

        if dist > MIN_MOVEMENT_THRESHOLD:
            if dist > MAX_SPEED:
                ratio = MAX_SPEED / dist
                current_x += dx * ratio
                current_y += dy * ratio
                current_x = int(current_x)
                current_y = int(current_y)
            else:
                current_x = target_x
                current_y = target_y

        x1_crop = int(current_x - output_width // 2)
        y1_crop = int(current_y - output_height // 2)
        x1_crop = max(0, min(x1_crop, w_frame - output_width))
        y1_crop = max(0, min(y1_crop, h_frame - output_height))
        crop_rects.append((x1_crop, y1_crop, output_width, output_height))

    return crop_rects

def main():
    if len(sys.argv) != 2:
        print("Usage: python reelmaker.py <input_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"The provided input '{input_dir}' is not a directory.")
        sys.exit(1)

    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    if not video_files:
        print(f"No .mp4 files found in directory '{input_dir}'.")
        sys.exit(1)

    output_dir = os.path.join(input_dir, 'processed_videos')
    os.makedirs(output_dir, exist_ok=True)

    for video_file in video_files:
        input_video = os.path.join(input_dir, video_file)
        output_video = os.path.join(output_dir, f"processed_{video_file}")
        debug_output_video = os.path.join(output_dir, f"debug_{video_file}")
        print(f"Processing {input_video}...")

        try:
            face_positions, motion_positions, fps, frame_dims = analyze_video(input_video)
            crop_rects = compute_crop_positions(face_positions, motion_positions, frame_dims, fps)

            clip = mpy.VideoFileClip(input_video)

            def process_frame(get_frame, t):
                frame_idx = int(t * fps)
                if frame_idx >= len(crop_rects):
                    frame_idx = len(crop_rects) - 1
                x1_crop, y1_crop, output_width, output_height = crop_rects[frame_idx]
                frame = get_frame(t)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cropped_frame = frame_bgr[y1_crop:y1_crop + output_height, x1_crop:x1_crop + output_width]
                resized_frame = cv2.resize(cropped_frame, (DESIRED_WIDTH, DESIRED_HEIGHT), interpolation=cv2.INTER_AREA)
                return cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            processed_clip = clip.fl(process_frame)
            processed_clip.write_videofile(output_video, codec="libx264", audio_codec="aac",
                                           temp_audiofile="temp-audio.m4a", remove_temp=True)

            if DEBUG:
                h_frame, w_frame = frame_dims
                top_limit = int(h_frame * TOP_MARGIN_PERCENT)
                bottom_limit = int(h_frame * (1 - BOTTOM_MARGIN_PERCENT))
                left_limit = int(w_frame * LEFT_MARGIN_PERCENT)
                right_limit = int(w_frame * (1 - RIGHT_MARGIN_PERCENT))

                def debug_process_frame(get_frame, t):
                    frame_idx = int(t * fps)
                    if frame_idx >= len(face_positions):
                        frame_idx = len(face_positions) - 1
                    frame = get_frame(t)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    draw_checkered_pattern(frame_bgr, (0,0), (w_frame, top_limit), (0,0,255))
                    draw_checkered_pattern(frame_bgr, (0,bottom_limit), (w_frame, h_frame), (0,0,255))
                    draw_checkered_pattern(frame_bgr, (0,0), (left_limit, h_frame), (0,0,255))
                    draw_checkered_pattern(frame_bgr, (right_limit,0), (w_frame, h_frame), (0,0,255))

                    if motion_positions[frame_idx] is not None:
                        x, y, w, h = motion_positions[frame_idx]
                        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0,255,0), 2)

                    if face_positions[frame_idx] is not None:
                        x1, y1, x2, y2 = face_positions[frame_idx]
                        cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (255,0,0), 2)

                    if frame_idx < len(crop_rects):
                        x1_crop, y1_crop, output_width, output_height = crop_rects[frame_idx]
                        cv2.rectangle(frame_bgr, (x1_crop, y1_crop), 
                                      (x1_crop + output_width, y1_crop + output_height), 
                                      (0,255,255), 3)

                    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                debug_clip = clip.fl(debug_process_frame)
                debug_clip.write_videofile(debug_output_video, codec="libx264", audio_codec="aac",
                                           temp_audiofile="temp-audio-debug.m4a", remove_temp=True)

        except Exception as e:
            print(f"An error occurred while processing {input_video}: {e}")

    print("Processing complete.")

def draw_checkered_pattern(img, start_pt, end_pt, color, square_size=20):
    x1, y1 = start_pt
    x2, y2 = end_pt
    for y in range(y1, y2, square_size):
        for x in range(x1, x2, square_size):
            if ((x//square_size) + (y//square_size)) % 2 == 0:
                cv2.rectangle(img, (x,y), (min(x2,x+square_size), min(y2,y+square_size)), color, -1)

if __name__ == "__main__":
    main()
