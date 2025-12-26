import streamlit as st
import subprocess
import os
from PIL import Image

# --- Page Config ---
st.set_page_config(
    page_title="Smart Object & Face Recognition",
    page_icon="🎯",
    layout="wide"
)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎯 Smart Real-Time Object & Face Recognition</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("🚀 Pipeline Options")
show_known_faces = st.sidebar.checkbox("Show Known Faces Folder")

# --- Step 1: Upload Image ---
st.subheader("Step 1️⃣ Upload a New Face Image")
uploaded_file = st.file_uploader("Upload an image of yourself", type=["jpg", "jpeg", "png"])
KNOWN_DIR = "known_faces"

if uploaded_file:
    os.makedirs(KNOWN_DIR, exist_ok=True)
    save_path = os.path.join(KNOWN_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved `{uploaded_file.name}` in `{KNOWN_DIR}/` folder")
    st.image(Image.open(save_path), caption="Uploaded Image", width = 300)

# --- Step 2: Generate/Update Embeddings ---
st.subheader("Step 2️⃣ Generate Embeddings")
if st.button("Generate/Update Embeddings"):
    GENERATE_SCRIPT = "src/generate_embedding.py"
    if os.path.exists(GENERATE_SCRIPT):
        try:
            st.info("Generating embeddings...")
            subprocess.run(["python", GENERATE_SCRIPT], check=True)
            st.success("✅ Embeddings generated successfully!")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Error generating embeddings:\n{e}")
    else:
        st.error(f"❌ `{GENERATE_SCRIPT}` not found!")

# --- Step 3: Run Real-Time Detection ---
st.subheader("Step 3️⃣ Run Real-Time Object & Face Detection")
if st.button("Start Real-Time Detection"):
    LAUNCHER_SCRIPT = "Launcher.py"
    if os.path.exists(LAUNCHER_SCRIPT):
        try:
            st.info("Launching real-time detection...")
            subprocess.run(["python", LAUNCHER_SCRIPT], check=True)
            st.success("✅ Detection finished!")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Error running real-time detection:\n{e}")
    else:
        st.error(f"❌ `{LAUNCHER_SCRIPT}` not found!")

# --- Show Known Faces ---
if show_known_faces and os.path.exists(KNOWN_DIR):
    st.subheader("👤 Known Faces")
    face_files = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if face_files:
        cols = st.columns(len(face_files))
        for i, file in enumerate(face_files):
            with cols[i]:
                img = Image.open(os.path.join(KNOWN_DIR, file))
                st.image(img, caption=file, width = 250)
    else:
        st.info("No images found in `known_faces/` folder.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Powered by YOLOv8 + FaceNet/ArcFace | Developed by Karamjodh Singh</p>", unsafe_allow_html=True)
