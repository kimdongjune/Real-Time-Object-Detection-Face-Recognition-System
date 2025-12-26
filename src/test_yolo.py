from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("data/test_image.jpg")
### YOLO takes the image "data/test_image.jpg" as input.Preprocessing happens: the image is resized, normalized, and converted into a tensor suitable for the model.Forward pass: the image tensor goes through the neural network.The network outputs predictions (bounding boxes, class probabilities, and confidence scores).###
for result in results:
    result.show()
