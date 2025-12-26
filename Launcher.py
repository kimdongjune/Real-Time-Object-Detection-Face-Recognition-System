# run_all.py
import subprocess
import os

# ---------------- CONFIG ----------------
SRC_DIR = "src"  # where your scripts are
GENERATE_SCRIPT = "generate_embedding.py"
REALTIME_SCRIPT = "realtime_detect.py"

# ---------------- STEP 1: Generate embeddings ----------------
print("Generating/updating embeddings...")
subprocess.run(["python", os.path.join(SRC_DIR, GENERATE_SCRIPT)], check=True)
print("Embeddings updated successfully!")

# ---------------- STEP 2: Launch real-time detection ----------------
print("Starting real-time detection...")
subprocess.run(["python", os.path.join(SRC_DIR, REALTIME_SCRIPT)], check=True)
