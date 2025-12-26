import cv2
from ultralytics import YOLO
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch

# Load known embeddings
data = torch.load("embeddings/known_embeddings.pt")
known_embeddings = data["embeddings"]
known_names = data["names"]

model = YOLO("model/yolov8n.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- YOLO OBJECT DETECTION ----------
    results = model(frame)
    res = results[0]

    frame = res.plot()   # draw YOLO boxes ON frame

    # ---------- FACE DETECTION ----------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        h, w, _ = frame.shape  # get frame dimensions
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # --------- SAFE BOX CLAMPING ----------
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Skip tiny boxes to avoid FaceNet crash
            if x2 - x1 < 50 or y2 - y1 < 50:
                continue

            # --------- SAFE FACE CROP & RESIZE ----------
            face = rgb_frame[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))  # FaceNet expects 160x160

            face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0)
            face_tensor = face_tensor.float().to(device) / 255.0

            with torch.no_grad():
                embedding = resnet(face_tensor)

            name = "Unknown"

            if len(known_embeddings) > 0:
                distances = [
                    torch.dist(embedding, known_emb).item()
                    for known_emb in known_embeddings
                ]
                min_dist = min(distances)

                if min_dist < 0.9:  # threshold
                    name = known_names[distances.index(min_dist)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )



    # ---------- DISPLAY ----------
    cv2.imshow("YOLO + Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
