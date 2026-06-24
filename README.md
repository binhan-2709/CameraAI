# Camera AI Attendance System Using Real-Time Face Recognition

This repository contains the complete source code for the **Camera AI Attendance System**, developed as part of an ICT Bachelor's Thesis at the **University of Science and Technology of Hanoi (USTH)** in collaboration with **Vinorsoft**.

The system utilizes standard webcams and deep learning models to automate employee check-in and check-out logs through face recognition, security liveness checks, and expression/gesture interaction gates.

---

## Key Features
- **Face Recognition:** Accurate detection and verification using YOLOv8-face, MTCNN, and InceptionResnetV1 (FaceNet).
- **Liveness Detection (Anti-Spoofing):** Rule-based 2D liveness filter checking texture frequency (Laplacian variance), contrast, HSV skin tone, overexposed glare, and phone-screen bezels.
- **Interaction Gates:**
  - *Check-in:* Smile-gated validation (happiness expression) using a lightweight Mini-Xception model.
  - *Check-out:* Gesture-gated validation (hand-waving detection) using absolute frame differencing and optical motion swings.
- **High-Performance Architecture:** Threaded camera frame acquisition and template-based ROI tracking (\texttt{cv2.matchTemplate}) to run at 25+ FPS on edge CPUs.
- **FastAPI Backend:** Asynchronous REST endpoints, SQLAlchemy ORM with SQLite, and WebSockets for real-time frame streaming and logging.
- **Web Dashboard:** Clean, responsive frontend with a split sidebar (green for Check-ins, orange for Check-outs) for administrators.

---

## Project Directory Structure
```text
data/
├── attendance.db         # Local SQLite database (SQLAlchemy logs)
├── settings.json         # Runtime config overrides
├── raw/                  # Enrolled face images grouped by employee ID
└── augmented/            # Synthesized face crops (Albumentations)
models/
├── emotion_ferplus.onnx  # Exported ONNX emotion classifier (Mini-Xception)
└── face_db.pkl           # Pickled reference embeddings database
src/
├── api/                  # FastAPI REST endpoints, WebSocket manager, ORM logic
├── data/                 # Face collection and augmentation scripts
├── detection/            # Face detection fallbacks and ONNX emotion class
├── pipeline/             # Multi-threaded camera frame acquisition and ROI tracking
└── recognition/          # Face Net embedding extraction and cosine matching
frontend/                 # Web dashboard files (HTML, CSS, JS)
tests/                    # Pytest suite for API and system verification
Dockerfile                # Image manifest for Docker container
docker-compose.yml        # Orchestration file for quick deployment
install.bat               # Windows local setup automation script
```

---

## Deployment Options

We provide two ways to run the project:
1. **Docker (Recommended for Supervisors / Quick Testing)** - Runs everything inside a pre-configured container. No Python installation required.
2. **Local Setup** - Running directly on your host machine in a Python virtual environment.

---

### Option 1: Running with Docker (Recommended)
This is the easiest way to inspect and test the project without manually configuring Python dependencies on your host system.

#### Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) on your system and ensure it is running.

#### Running the Container
1. Open a terminal (Command Prompt, PowerShell, or Bash) in the project root directory.
2. Start the services using Docker Compose:
   ```bash
   docker-compose up --build
   ```
3. Docker will automatically pull the Python base image, install system dependencies (OpenCV GL libraries), install all Python packages, mount your local data folders, and start the backend service.
4. **Access the Dashboard:** Open your browser and navigate to:
   ```
   http://localhost:8000
   ```
   *Note: FastAPI serves the dashboard statically on the root path `/` inside the container. Interactive Swagger API documentation is available at `http://localhost:8000/docs`.*
5. **Stop the Container:** Press `Ctrl+C` in the terminal, or run:
   ```bash
   docker-compose down
   ```

---

### Option 2: Local Setup (Manual)
#### Prerequisites
- Install **Python 3.10** or **3.11**.
- Install **Git**.

#### Installation Steps
1. Open your terminal in the project directory.
2. **Automatic Setup (Windows):**
   Double-click the `install.bat` file, or run:
   ```cmd
   install.bat
   ```
3. **Manual Setup (All Operating Systems):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   # On Windows CPU only (recommended to avoid large CUDA downloads):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

#### Usage Workflow
1. **Collect Registration Faces:**
   Run the face collection tool to register employee profiles (captures or uploads):
   ```bash
   python src/data/collect_faces.py
   ```
2. **Apply Data Augmentation:**
   Generate synthetic variations for lighting/pose robustness:
   ```bash
   python src/data/augment.py
   ```
3. **Build the Face Embedding Database:**
   Extract and compile reference embeddings:
   ```bash
   python src/recognition/build_database.py
   ```
4. **Start the FastAPI Backend:**
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```
5. **Start the Frontend Server:**
   ```bash
   python -m http.server 5500 --directory frontend
   ```
6. **Open the Dashboard:** Navigate to `http://localhost:5500` in your web browser. (The dashboard calls the API running on `http://127.0.0.1:8000`).
7. **Run the Real-time Camera Pipeline:**
   To run the webcam feed with live visual annotations:
   ```bash
   python src/pipeline/realtime_pipeline.py
   ```

---

## Configuration (`.env`)
Create a `.env` file in the root directory (based on `.env` example):
- `CAMERA_SOURCE`: Camera index (e.g. `0` for default webcam, or a video file path/RTSP stream URL).
- `FACE_BACKEND`: Detector selection (`auto`, `yolov8`, `mtcnn`, `haar`). `auto` falls back automatically.
- `FACE_THRESHOLD`: Cosine similarity threshold (recommended `0.65`).
- `CHECKIN_COOLDOWN_MINUTES`: Cooldown period to prevent duplicate check-ins.
- `LATE_THRESHOLD`: Arrival limit (e.g. `08:30`).

---

## Running Tests
To run the automated test suite and verify code integrity:
```bash
pytest tests/ -v
```
