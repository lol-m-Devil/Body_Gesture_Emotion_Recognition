import cv2
import mediapipe as mp
import numpy as np

from scipy.sparse import coo_matrix


# Initialize mediapipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load input video
video_path = 'data/t2.mp4'
cap = cv2.VideoCapture(video_path)

# Create or open a text file to store pose coordinates
output_file = open('pose_coordinates.txt', 'w')
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
        for coord in landmarks:
            output_file.write(f"{coord[0]} {coord[1]}\n")
        output_file.write("\n")  # Separate each frame's coordinates with an empty line
        
        # Draw landmarks on the frame
        for landmark in results.pose_landmarks.landmark:
            height, width, _ = frame.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Draw a filled circle
    
    # Display frame
    # cv2.imshow('Pose Detection', frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Close the output file
output_file.close()


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


# Assuming you have a list of 144 frames, each containing 33 (x, y) coordinates normalized from 0 to 1
num_frames = len(pose_coords)
num_coords = pose_coords[0].shape[0]  # Assuming all frames have the same number of coordinates

# Define the size of the output sparse matrices
matrix_size = 64

# Initialize a list to store sparse matrices for each frame
sparse_matrices = []

# Iterate over each frame
for frame in pose_coords:
    # Initialize coordinate lists
    row_indices = []
    col_indices = []
    data_values = []
    
    # Iterate over each (x, y) coordinate in the frame
    for coord in frame:
        # Scale the (x, y) coordinates to the range [0, 63]
        scaled_x = int(coord[0] * (matrix_size - 1))
        scaled_y = int(coord[1] * (matrix_size - 1))
        
        # Convert (x, y) coordinates to row and column indices in the sparse matrix
        row_index = scaled_y
        col_index = scaled_x
        
        # Append the row and column indices, and the data value (1) to the lists
        row_indices.append(row_index)
        col_indices.append(col_index)
        data_values.append(1)
    
    # Create the sparse matrix for the current frame
    sparse_matrix = coo_matrix((data_values, (row_indices, col_indices)), shape=(matrix_size, matrix_size))
    
    # Append the sparse matrix to the list
    sparse_matrices.append(sparse_matrix)



