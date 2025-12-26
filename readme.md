Real-Time Object Detection and Face Recognition

This project is an end-to-end real-time Object Detection and Face Recognition system built using modern computer vision and deep learning techniques. It combines YOLOv8 for object detection, MTCNN for face detection, and FaceNet (InceptionResnetV1) for face recognition. The system works on a live webcam feed and can identify known individuals based on previously generated facial embeddings.

The workflow is designed to be user-friendly. First, users upload their face images into the known_faces folder. Then facial embeddings are generated and stored locally. Finally, real-time detection is performed where faces are detected and matched against the stored embeddings. Object detection runs in parallel to identify common objects in the scene.

The project also includes a Streamlit-based web interface that allows users to upload images, generate embeddings, and start real-time face recognition without manually running multiple scripts. A launcher script is used internally to manage the execution of different components in sequence.

Project Structure Overview:
The project contains folders for data samples, model weights, known face images, generated embeddings, and source code scripts. The models folder stores YOLO weights, while embeddings are stored locally and ignored by Git to ensure privacy.

Key Features:
Real-time object detection using YOLOv8
Accurate face detection using MTCNN
Face recognition using deep face embeddings
Live webcam-based recognition
Streamlit UI for ease of use
Privacy-aware design with local data storage

Technologies Used:
Python, OpenCV, PyTorch, facenet-pytorch, YOLOv8 (Ultralytics), Streamlit

Privacy and Security:
Personal face images and embeddings are generated and stored locally on the user’s machine. These files are excluded from version control to prevent accidental data leaks. Users have full control over their data.

Future Enhancements:
Potential improvements include face registration via webcam, multi-user support, database integration, performance optimization, and deployment on edge devices or cloud platforms.

Author:
Karamjodh Singh