import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Paths
KNOWN_FACES_DIR = "known_faces"
EMBEDDINGS_PATH = "embeddings/known_embeddings.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize MTCNN and ResNet
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

known_embeddings = []
known_names = []

# Loop through known faces folder
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        name = filename.split('.')[0]
        path = os.path.join(KNOWN_FACES_DIR, filename)

        img = Image.open(path)
        face_tensor = mtcnn(img)
        if face_tensor is None:
            print(f"No face detected in {filename}, skipping.")
            continue

        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face_tensor)

        known_embeddings.append(embedding)
        known_names.append(name)
        print(f"Processed {name}")

# Stack embeddings and save
known_embeddings = torch.cat(known_embeddings)
torch.save({'embeddings': known_embeddings, 'names': known_names}, EMBEDDINGS_PATH)
print(f"Saved embeddings to {EMBEDDINGS_PATH}")
