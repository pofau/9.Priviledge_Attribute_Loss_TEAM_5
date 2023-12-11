import cv2
import numpy as np
import face_recognition


def heatmap_generator(image):
    face_locations = face_recognition.face_locations(image)

    # Load the pre-trained facial landmark model
    face_landmarks_list = face_recognition.face_landmarks(image, face_locations)

    h,w = image.shape[:2]
    lm = np.zeros([h,w])

    # Draw facial landmarks on the image
    for face_landmarks in face_landmarks_list:
        for landmark_type, landmarks in face_landmarks.items():
            for (x, y) in landmarks:
                if x < h and y < w :
                    lm[y,x] = 1

    heatmap = cv2.GaussianBlur(lm, [59,59], 3)         

    return heatmap

def generate_batch_heatmaps(images, heatmap_generator):
    batch_heatmaps = torch.zeros_like(images)

    for i in range(images.size(0)):
        # Convertir le tenseur PyTorch en tableau NumPy pour l'image i
        image_np = images[i].permute(1, 2, 0).cpu().detach().numpy()
        image_np = (image_np * 255).astype(np.uint8) if image_np.dtype != np.uint8 else image_np
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
        # Générer la heatmap pour l'image actuelle
        heatmap_np = heatmap_generator(image_np)
        heatmap_tensor = torch.from_numpy(heatmap_np).float().unsqueeze(0)

        # Normaliser la heatmap et l'adapter à la taille de l'image
        heatmap_tensor = heatmap_tensor / torch.max(heatmap_tensor)
        heatmap_tensor = heatmap_tensor.repeat(3, 1, 1)

        # Stocker la heatmap dans le tenseur batch
        batch_heatmaps[i] = heatmap_tensor
    
    return batch_heatmaps
