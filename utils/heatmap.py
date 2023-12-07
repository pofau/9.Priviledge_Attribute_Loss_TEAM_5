import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_landmarks(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the dlib face detector and facial landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    

    # Detect faces in the image
    faces = detector(gray)
    
    # Get the facial landmarks
    for face in faces:
        landmarks = predictor(gray, face)
    
    # Draw the facial landmarks
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
    
    return image, landmarks

def heatmap_generation(image,landmarks):
    
    # The image dimensions
    height, width = image.shape[:2]
    
    # Create a blank heatmap with the same dimensions as the original image
    heatmap = np.zeros([height,width])
    print(heatmap.shape)
    # Draw the landmarks on the heatmap
    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        # Set pixel value to 1 at landmark position
        heatmap[y, x] = 1

    # Apply gaussian filter
    filtered_heatmap = dlib.gaussian_blur(heatmap,3)[0]

    return filtered_heatmap
    
# Testing the functions

# # The image path
# image_path = "./example.jpg"

# # Read the image
# image = cv2.imread(image_path)
# # Landmarks detection
# result, landmarks = detect_landmarks(image)

# # Display the image with landmarks
# # cv2.imshow("Facial Landmarks", result)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # Heatmap generation
# heatmap = heatmap_generation(image, landmarks)

# # Display the heatmap
# plt.imshow(heatmap, cmap=plt.cm.gray)