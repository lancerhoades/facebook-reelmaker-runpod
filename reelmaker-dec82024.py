import cv2
import numpy as np
import moviepy.editor as mpy
import sys
import os

# Update BASE_DIR to the absolute path
BASE_DIR = r"C:\Users\lance\coding\facebookreelmaker"

# Update MODEL_FILE and CONFIG_FILE to use BASE_DIR
MODEL_FILE = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
CONFIG_FILE = os.path.join(BASE_DIR, "deploy.prototxt")


# Desired output video dimensions (for 9:16 aspect ratio)
DESIRED_WIDTH = 1080
DESIRED_HEIGHT = 1920

def load_face_detector():
    """
    Loads the DNN face detection model.
    """
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    
    net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)
    return net

def detect_faces_dnn(net, frame):
    """
    Detects faces in a frame using the DNN model.
    """
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
        if confidence > 0.6:  # Adjust confidence threshold if necessary
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faces.append([x1, y1, x2, y2])
    return faces

def compute_motion_between_frames(prev_gray, curr_gray):
    """
    Computes motion between two consecutive grayscale frames.
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return thresh

def analyze_video(input_video):
    """
    Analyzes the input video to detect face positions and motion.
    """
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
        
        # Detect faces
        faces = detect_faces_dnn(net, frame_bgr)
        if faces:
            largest_face = max(faces, key=lambda rect: (rect[2]-rect[0]) * (rect[3]-rect[1]))
            face_positions.append(largest_face)
        else:
            face_positions.append(None)
        
        # Compute motion
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
    """
    Computes crop positions for each frame based on detected faces and motion.
    """
    h_frame, w_frame = frame_dims
    centers = []
    for face_bbox, motion_bbox in zip(face_positions, motion_positions):
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            face_center_x = x1 + (x2 - x1) // 2
            face_center_y = y1 + (y2 - y1) // 2
            centers.append((face_center_x, face_center_y))
        elif motion_bbox is not None:
            x_motion, y_motion, w_motion, h_motion = motion_bbox
            motion_center_x = x_motion + w_motion // 2
            motion_center_y = y_motion + h_motion // 2
            centers.append((motion_center_x, motion_center_y))
        else:
            centers.append(None)

    for i in range(len(centers)):
        if centers[i] is None:
            if i > 0:
                centers[i] = centers[i - 1]
            else:
                centers[i] = (w_frame // 2, h_frame // 2)

    frames_per_segment = int(fps)
    num_segments = int(np.ceil(len(centers) / frames_per_segment))
    segment_centers = []

    for i in range(num_segments):
        start_idx = i * frames_per_segment
        end_idx = min((i + 1) * frames_per_segment, len(centers))
        segment = centers[start_idx:end_idx]
        avg_x = int(np.mean([c[0] for c in segment]))
        avg_y = int(np.mean([c[1] for c in segment]))
        segment_centers.append((avg_x, avg_y))

    crop_rects = []
    output_width = int(h_frame * output_aspect_ratio)
    output_height = h_frame

    if output_width > w_frame:
        output_width = w_frame
        output_height = int(w_frame / output_aspect_ratio)

    total_frames = len(centers)
    for i in range(total_frames):
        segment_idx = i // frames_per_segment
        t = (i % frames_per_segment) / frames_per_segment

        current_center = segment_centers[segment_idx]
        if segment_idx + 1 < len(segment_centers):
            next_center = segment_centers[segment_idx + 1]
        else:
            next_center = current_center

        center_x = int((1 - t) * current_center[0] + t * next_center[0])
        center_y = int((1 - t) * current_center[1] + t * next_center[1])

        x1_crop = int(center_x - output_width // 2)
        y1_crop = int(center_y - output_height // 2)

        x1_crop = max(0, min(x1_crop, w_frame - output_width))
        y1_crop = max(0, min(y1_crop, h_frame - output_height))

        crop_rects.append((x1_crop, y1_crop, output_width, output_height))

    return crop_rects

def main():
    """
    Main function to process videos in the input directory.
    """
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
                cropped_frame = frame_bgr[y1_crop : y1_crop + output_height, x1_crop : x1_crop + output_width]
                resized_frame = cv2.resize(cropped_frame, (DESIRED_WIDTH, DESIRED_HEIGHT), interpolation=cv2.INTER_AREA)
                return cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            processed_clip = clip.fl(process_frame)
            processed_clip.write_videofile(output_video, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a", remove_temp=True)
        except Exception as e:
            print(f"An error occurred while processing {input_video}: {e}")

    print("Processing complete.")

if __name__ == "__main__":
    main()
