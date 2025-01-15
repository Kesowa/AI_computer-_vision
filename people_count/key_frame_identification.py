import cv2
import numpy as np

def identify_keyframes(video_path, threshold=30, min_change_area=500):
    """
    Identify keyframes in a video based on changes in the scene.

    Parameters:
        video_path (str): Path to the video file.
        threshold (int): Threshold for detecting frame differences (default: 30).
        min_change_area (int): Minimum area of change to consider as a keyframe (default: 500).

    Returns:
        list: A list of frame indices considered as keyframes.
    """
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    prev_frame = None
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        if prev_frame is not None:
            # Compute absolute difference between current and previous frames
            frame_diff = cv2.absdiff(prev_frame, gray_frame)
            _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours of the changes
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_change_area = sum(cv2.contourArea(c) for c in contours)
            
            # Check if the change area exceeds the minimum threshold
            if total_change_area > min_change_area:
                keyframes.append(frame_index)
        
        prev_frame = gray_frame
        frame_index += 1

    cap.release()
    return keyframes

# Example usage
video_file = 'path_to_your_video.mp4'
keyframes = identify_keyframes(video_file)
print(f"Identified keyframes: {keyframes}")
