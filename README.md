# PrivAIMin

## Private AI Attendance Management System

PrivAIMin is an AI-powered attendance management system developed as a capstone project at the British University in Dubai (BUID). The system automates attendance tracking using facial recognition, vector similarity search, and a user-friendly web interface.

The project combines MTCNN face detection, FaceNet deep facial embeddings, Qdrant vector database technology, and Streamlit to provide a complete attendance management solution capable of recognizing individuals from uploaded images, live camera captures, and group photographs.

Unlike traditional attendance systems, PrivAIMin stores facial embeddings rather than relying on direct image comparison, improving scalability, efficiency, and privacy.

---

# Authors

Fatma Zainal
Rand Abubaker
Shaima Abuserdaneh

British University in Dubai (BUID)

Supervisor: Dr. Ahmed Awad

Capstone Project 2026

---

# Project Features

## Student Enrollment

* Upload student facial images
* Generate FaceNet embeddings
* Create centroid embeddings from multiple images
* Store student records in Qdrant

## Face Preprocessing

* MTCNN face detection
* Facial alignment using landmarks
* White background enhancement
* Face normalization
* Image resizing to 160×160

## Face Recognition

* Single-face recognition
* Group-face recognition
* Top-K similarity search
* Threshold-based decision making
* Unknown face detection

## Attendance Management

* Automatic attendance logging
* Attendance history tracking
* CSV attendance export
* Timestamp recording

## Analytics Dashboard

* Attendance statistics
* Recognition performance metrics
* Daily attendance summaries
* Attendance trends

## Manual Verification

* Review recognition results
* Verify unknown detections
* Improve recognition reliability

---

# System Architecture

```text
Dataset Images
      │
      ▼
Face Detection (MTCNN)
      │
      ▼
Face Alignment
      │
      ▼
White Background Processing
      │
      ▼
FaceNet Embedding Generation
      │
      ▼
Centroid Embedding Creation
      │
      ▼
Qdrant Vector Database
      │
      ▼
Similarity Search
      │
      ▼
Decision Logic
      │
      ▼
Attendance Logging
```

---

# Technologies Used

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Core development        |
| Streamlit    | Web application         |
| MTCNN        | Face detection          |
| FaceNet      | Facial recognition      |
| Qdrant       | Vector database         |
| OpenCV       | Image processing        |
| PyTorch      | Deep learning framework |
| NumPy        | Numerical computations  |
| Pandas       | Data analysis           |
| Scikit-Learn | Evaluation metrics      |

---

# Repository Structure

```text
capstone/
│
├── app.py
├── extract_clean_faces.py
├── savetoFaceDB.py
├── recognize_group.py
├── queryFaceDB.py
├── train_embeddings_threshold.py
├── evaluate_metrics.py
├── inspect_embeddings.py
├── check_qdrant_dataset.py
│
├── my_dataset/
├── clean_faces_4_white/
├── attendance_captures/
├── attendance_log.csv
│
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/1fatmayz/capstone.git
cd capstone
```

## Create Virtual Environment

```bash
python -m venv .venv
```

## Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install streamlit
pip install opencv-python
pip install numpy
pip install pandas
pip install torch
pip install facenet-pytorch
pip install qdrant-client
pip install pillow
pip install scikit-learn
```

---

# Qdrant Setup

Run Qdrant locally using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Verify that the database is running:

```text
http://localhost:6333/dashboard
```

---

# Preparing the Dataset

Create a dataset folder with one folder per student.

Example:

```text
my_dataset/
│
├── Student_01/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
├── Student_02/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
```

---

# Step 1 – Face Extraction and Preprocessing

Run:

```bash
python extract_clean_faces.py
```

This script:

* Detects faces using MTCNN
* Aligns faces using eye landmarks
* Applies white background enhancement
* Resizes faces to 160×160
* Saves processed faces to:

```text
clean_faces_4_white/
```

---

# Step 2 – Enroll Students into Qdrant

Run:

```bash
python savetoFaceDB.py
```

This script:

* Loads processed facial images
* Generates FaceNet embeddings
* Creates centroid embeddings
* Stores vectors in Qdrant

Collection name:

```text
tryface
```

---

# Step 3 – Inspect Stored Data

Run:

```bash
python check_qdrant_dataset.py
```

This script displays:

* Collection information
* Enrolled students
* Payload metadata
* Stored records

---

# Step 4 – Test Recognition Queries

Run:

```bash
python queryFaceDB.py
```

This script:

* Generates embeddings from query images
* Searches Qdrant
* Returns Top-K matches
* Applies recognition thresholds

Default settings:

```text
TOP_K = 5
Threshold = 0.72
Gap = 0.05
```

---

# Step 5 – Group Recognition

Run:

```bash
python recognize_group.py
```

This script:

* Detects multiple faces in a single image
* Generates embeddings for each face
* Searches Qdrant
* Labels known faces
* Marks unknown faces
* Saves attendance events

---

# Step 6 – Evaluate System Performance

Run:

```bash
python evaluate_metrics.py
```

The script calculates:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# Step 7 – Threshold Optimization

Run:

```bash
python train_embeddings_threshold.py
```

This script:

* Generates embeddings
* Creates positive and negative comparison pairs
* Evaluates threshold ranges
* Finds optimal recognition thresholds

---

# Launching the Web Application

Before running the web application, make sure Qdrant is running.

Then run:

```bash
streamlit run app.py
```

The application will open on the computer at:

```text
http://localhost:8501
```

---

# Web Application Pages

## Dashboard

Displays:

* Attendance statistics
* Recognition summaries
* Recent attendance activity

## Live Recognition

Allows:

* Camera capture
* Image upload
* Real-time recognition
* Attendance registration

## Analytics

Displays:

* Attendance trends
* Daily attendance percentages
* Recognition statistics

## Records

Displays:

* Attendance logs
* Searchable attendance records
* CSV export functionality

## Manual Verification

Allows:

* Reviewing recognition results
* Managing unknown detections

## Settings

Allows:

* Threshold configuration
* Top-K configuration
* Qdrant connection settings

---

# Mobile Access

To access the Streamlit app on another device, use the ngrok public link.

First, run the Streamlit app:

```bash
streamlit run app.py
```

The Streamlit terminal should show something similar to:

```text
Local URL: http://localhost:8501
Network URL: http://172.22.1.2:8501
```

Then open another PowerShell or terminal window and run:

```bash
ngrok http 8501
```

Ngrok will generate a forwarding link similar to:

```text
https://showroom-nuclear-qualifier.ngrok-free.dev
```

Open the ngrok forwarding link on your phone:

```text
https://showroom-nuclear-qualifier.ngrok-free.dev
```

Do not use `localhost` on the phone because `localhost` only works on the computer running the app.

---

# Important Notes for Running the Project

* `app.py` must be run from the correct project folder.
* If PowerShell says `File does not exist: app.py`, move into the project folder first.

Example:

```bash
cd "C:\Users\ffatm\OneDrive\Desktop\capstone\Attendance-Face-Detection"
streamlit run app.py
```

* Keep the Streamlit terminal open while using the website.
* Keep the ngrok terminal open while accessing the website from phone.
* Keep Qdrant running before recognition or enrollment.

---

# Public Access with Ngrok

Run:

```bash
ngrok http 8501
```

Example:

```text
https://example.ngrok-free.app
```

Share this link to access the application remotely.

---

# Performance Results

Evaluation dataset:

* Total Test Images: 760
* Correct Recognitions: 711
* Unknown Detections: 49

Performance metrics:

| Metric    | Result  |
| --------- | ------- |
| Accuracy  | 93.55%  |
| Precision | 100.00% |
| Recall    | 93.55%  |
| F1 Score  | 96.60%  |

---

# Privacy and Security

PrivAIMin prioritizes privacy by:

* Using facial embeddings instead of direct image comparison
* Storing vector representations in Qdrant
* Performing recognition locally
* Reducing exposure of raw biometric data

---

# Future Improvements

* Anti-spoofing protection
* Cloud deployment
* Multi-camera support
* Mobile application integration
* Advanced attendance analytics
* Larger-scale evaluation datasets

---

# Repository

GitHub Repository:

```text
https://github.com/1fatmayz/capstone
```

---

# Acknowledgements

The authors would like to thank Dr. Ahmed Awad and the British University in Dubai (BUID) for their guidance, support, and supervision throughout the development of this capstone project.

---

© 2026 Fatma Zainal, Rand Abubaker, Shaima Abuserdaneh

British University in Dubai (BUID)
