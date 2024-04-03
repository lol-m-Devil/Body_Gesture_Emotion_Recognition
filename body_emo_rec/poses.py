import cv2
import mediapipe as mp
import numpy as np
import torch

def bodyJoints(videoPath):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(videoPath)

    pose_coords = []
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        # Detect pose landmarks
        results = pose.process(frame_rgb)
    
        # Extract pose coordinates
        if results.pose_landmarks:
            # Extract x, y coordinates of 33 body joints
            landmarks = np.array([[landmark.x, landmark.y] for landmark in results.pose_landmarks.landmark])
            # Write coordinates to the text file
            pose_coords.append(landmarks)
            
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    # stored in pose_coords dimension is (# of frames)*33*2
    pose_coords = np.array(pose_coords)
    # Find the maximum and minimum values across all coordinates
    for i in range(len(pose_coords)):
        max_val_x = np.max(pose_coords[i, :, 0])
        min_val_x = np.min(pose_coords[i, :, 0])
        
        max_val_y = np.max(pose_coords[i, :, 1])
        min_val_y = np.min(pose_coords[i, :, 1])
        
        pose_coords[i, :, 0] = (pose_coords[i, :, 0] - min_val_x) / (max_val_x - min_val_x)
        pose_coords[i, :, 1] = (pose_coords[i, :, 1] - min_val_y) / (max_val_y - min_val_y)

    pose_coords = pose_coords.astype(np.float32)
    return torch.tensor(pose_coords)

# poseCoords = bodyJoints('body_emo_rec/data/training/01-02-03-02-02-02-01.mp4')
# print(poseCoords.shape)
# print(type(poseCoords))
# print(poseCoords)