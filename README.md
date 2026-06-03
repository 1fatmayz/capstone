# PrivAIMin

## Private AI Attendance Management System

PrivAIMin is an AI-powered attendance management system developed as a Capstone Project at the British University in Dubai (BUID). The system automates attendance tracking using facial recognition, vector similarity search, and a user-friendly web interface.

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

# Overview

PrivAIMin automates attendance recording using facial recognition and vector database retrieval. The system detects faces, generates facial embeddings, compares them against enrolled identities stored in Qdrant, and records attendance automatically.

The solution is designed to provide accurate recognition, efficient retrieval, and a scalable architecture suitable for educational and organizational environments.

---

# Features

### Student Enrollment

* Upload facial images
* Generate FaceNet embeddings
* Create centroid embeddings from multiple images
* Store enrolled identities in Qdrant

### Face Preprocessing

* Face detection using MTCNN
* Face alignment using facial landmarks
* White background enhancement
* Image normalization
* Face resizing to 160×160 pixels

### Face Recognition

* Single-face recognition
* Group-face recognition
* Unknown face detection
* Similarity-based matching
* Top-K retrieval from Qdrant

### Attendance Management

* Automatic attendance logging
* Attendance history tracking
* Timestamp recording
* Attendance record management

### Analytics Dashboard

* Attendance summaries
* Recognition statistics
* Attendance trends
* Performance monitoring

### Manual Verification

* Review recognition results
* Verify uncertain matches
* Improve attendance reliability

---

# Technologies Used

| Technology                  | Purpose                 |
| --------------------------- | ----------------------- |
| Python                      | Core development        |
| Streamlit                   | Web application         |
| OpenCV                      | Image processing        |
| MTCNN                       | Face detection          |
| FaceNet (InceptionResnetV1) | Face recognition        |
| PyTorch                     | Deep learning framework |
| Qdrant                      | Vector database         |
| NumPy                       | Numerical computations  |
| Pandas                      | Data analysis           |
| Scikit-Learn                | Evaluation metrics      |

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
Face Preprocessing
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
Recognition Decision Logic
      │
      ▼
Attendance Logging
```

---

# Face Recognition Pipeline

### 1. Face Detection

Faces are detected using the MTCNN deep learning model.

### 2. Face Alignment and Preprocessing

Detected faces are aligned, normalized, enhanced with a white background, and resized to 160×160 pixels.

### 3. Face Embedding Generation

FaceNet generates a 512-dimensional embedding vector representing each face.

### 4. Enrollment

Multiple embeddings belonging to the same person are averaged to create a centroid embedding. The centroid is stored in Qdrant together with identity metadata.

### 5. Recognition

A query image is converted into an embedding and compared against stored vectors using cosine similarity.

### 6. Decision Logic

Recognition decisions are based on:

```text
Top K = 5
Recognition Threshold = 0.60
Gap Threshold = 0.05
```

---

# Repository Structure

```text
Attendance-Face-Detection/
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
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Installation

### Clone Repository

```bash
git clone https://github.com/1fatmayz/capstone.git
cd Attendance-Face-Detection
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Qdrant Setup

Run Qdrant locally using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Verify that Qdrant is running:

```text
http://localhost:6333/dashboard
```

Default collection:

```text
tryface
```

---

# Running the System

### Step 1 – Face Preprocessing

```bash
python extract_clean_faces.py
```

### Step 2 – Enroll Faces into Qdrant

```bash
python savetoFaceDB.py
```

### Step 3 – Launch the Web Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# Web Application Modules

## Dashboard

Displays attendance summaries, statistics, and recognition activity.

## Live Recognition

Recognizes individuals from uploaded images or camera captures.

## Analytics

Displays attendance and recognition performance metrics.

## Records

Displays attendance history and attendance logs.

## Manual Verification

Allows review and verification of recognition results.

## Enroll Student

Allows enrollment of new identities into the vector database.

## Settings

Allows adjustment of recognition parameters and system configuration.

---

# Mobile Access

Run Streamlit:

```bash
streamlit run app.py
```

Run ngrok in a separate terminal:

```bash
ngrok http 8501
```

Example public URL:

```text
https://showroom-nuclear-qualifier.ngrok-free.dev
```

Use the generated URL on any device with internet access.

---

# Dataset

The project supports datasets organized in person-specific folders.

Example:

```text
my_dataset/
├── Ahmed/
├── Fatma/
├── Shaima/
└── ...
```

The preprocessing pipeline generates aligned facial images that are used for enrollment and recognition.

The project also supports public face datasets and Kaggle face datasets for testing and evaluation purposes.

---

# Evaluation Metrics

The system was evaluated using a facial recognition test dataset.

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 93.55%  |
| Precision | 100.00% |
| Recall    | 93.55%  |
| F1 Score  | 96.60%  |

---

# Latency and Speed Results

Performance testing was conducted to measure the efficiency of the recognition pipeline.

| Performance Metric          | Result         |
| --------------------------- | -------------- |
| Average Embedding Latency   | 274.75 ms      |
| Average Recognition Latency | 0.25 ms        |
| Average Total Latency       | 275.01 ms      |
| Minimum Total Latency       | 122.57 ms      |
| Maximum Total Latency       | 839.47 ms      |
| Recognition Speed           | 3.64 faces/sec |

These results demonstrate that the system is capable of near real-time recognition while maintaining high recognition accuracy.

---

# Privacy and Security

PrivAIMin prioritizes privacy by:

* Using facial embeddings instead of direct image comparison
* Storing vector representations rather than raw biometric templates
* Supporting local deployment
* Reducing exposure of sensitive facial data

---

# Future Work

* Anti-spoofing detection
* Cloud deployment
* Mobile application integration
* Multi-camera recognition
* Advanced attendance analytics
* Larger-scale evaluation datasets

---

# License

This project was developed for academic and research purposes as part of the BUID Capstone Project 2026.

---

# Acknowledgements

The authors would like to thank Dr. Ahmed Awad and the British University in Dubai (BUID) for their guidance and support throughout the development of this project.

---

© 2026 Fatma Zainal, Rand Abubaker, Shaima Abuserdaneh

British University in Dubai (BUID)
