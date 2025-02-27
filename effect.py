import cv2
import mediapipe as mp
import numpy as np
import math

WIDTH = 1280
HEIGHT = 720
OFFSET = 90

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(1)

# Helper function to calculate angle between joints
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

# Helper function to calculate distance between joints
def calculate_distance(a, b):
    a = np.multiply(a, [WIDTH, HEIGHT]).astype(int)
    b = np.multiply(b, [WIDTH, HEIGHT]).astype(int)
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# Function to draw chakra ball
def draw_chakra_ball(image, position, offset):
    # Calculate the coordinates for drawing
    center = np.multiply(position, [WIDTH, HEIGHT]).astype(int)
    center[0] += offset  
    center = tuple(center)
    
    # Draw a chakra ball (a glowing orb with a halo effect)
    for radius, alpha in zip([40, 50, 60], [0.4, 0.3, 0.2]):
        overlay = image.copy()
        cv2.circle(overlay, center, radius, (0, 255, 255), thickness=-1)  # Outer halo 
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Inner core 
    cv2.circle(image, center, 30, (0, 255, 255), thickness=-1)  


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angles
            left_shoulder_angle = calculate_angle(right_shoulder, left_shoulder, left_elbow)
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_shoulder_angle = calculate_angle(left_shoulder, right_shoulder, right_elbow)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Calculate wrist distance
            wrist_dist = calculate_distance(left_wrist, right_wrist)
            
            # Visualize joints
            # cv2.putText(image, 'right shoulder', 
            #                tuple(np.multiply(right_shoulder, [WIDTH, HEIGHT]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # cv2.putText(image, 'right elbow', 
            #                tuple(np.multiply(right_elbow, [WIDTH, HEIGHT]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # cv2.putText(image, 'right wrist', 
            #                tuple(np.multiply(right_wrist, [WIDTH, HEIGHT]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # cv2.putText(image, 'left shoulder', 
            #                tuple(np.multiply(left_shoulder, [WIDTH, HEIGHT]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # cv2.putText(image, 'left elbow', 
            #                tuple(np.multiply(left_elbow, [WIDTH, HEIGHT]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # cv2.putText(image, 'left wrist', 
            #                tuple(np.multiply(left_wrist, [WIDTH, HEIGHT]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            
            # Drawing logic
            if left_shoulder_angle >= 120 and left_elbow_angle >= 160:
                draw_chakra_ball(image, left_wrist, OFFSET)  
            if right_shoulder_angle >= 120 and right_elbow_angle >= 160:
                draw_chakra_ball(image, right_wrist, -OFFSET)
            if wrist_dist <= 50:
                draw_chakra_ball(image, right_wrist, wrist_dist / 2)
                       
        except:
            pass
        
        # Render detections
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
        #                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
        #                          )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
