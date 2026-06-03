# PrivAIMin

## Private AI Attendance Management System

PrivAIMin is an AI-powered attendance management system developed as a capstone project at the British University in Dubai (BUID). The system automates attendance tracking using facial recognition, vector similarity search, and a user-friendly Streamlit web interface.

The project combines MTCNN face detection, FaceNet deep facial embeddings, Qdrant vector database technology, and Streamlit to recognize individuals from uploaded images, live camera captures, and group photographs.

Instead of comparing raw images directly, PrivAIMin converts faces into 512-dimensional embeddings and stores them in a vector database. This improves recognition speed, scalability, and privacy.

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
* Detect and preprocess faces
* Generate FaceNet embeddings
* Create centroid embeddings from multiple images
* Store student records in Qdrant

## Face Preprocessing

* Face detection using MTCNN
* Facial alignment using landmarks
* White background enhancement
* Face normalization
* Image resizing to 160×160 pixels

## Face Recognition

* Single-face recognition
* Group-face recognition
* Top-K similarity search
* Threshold-based decision making
* Unknown face detection

## Attendance Management

* Automatic attendance logging
* Attendance history tracking
* Timestamp recording
* CSV attendance export

## Dashboard and Analytics

* Attendance statistics
* Recognition performance metrics
* Daily attendance summaries
* Attendance trends

## Manual Verification

* Review recognition results
* Verify unknown detections
* Improve attendance reliability

---

# System Architecture

```text
Dataset Images
      │
      ▼
Face Detection using MTCNN
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
Recognition Decision Logic
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
| NumPy        | Numerical computation   |
| Pandas       | Data analysis           |
| Scikit-Learn | Evaluation metrics      |

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
├── my_dataset/
│   ├── Ahmed/
│   ├── Fatma/
│   ├── FYZ/
│   ├── Khaled Shaalan/
│   ├── khawla/
│   ├── Randa/
│   └── Shaima/
│
├── clean_faces_4_white/
│   ├── Ahmed/
│   ├── Akshay Kumar/
│   ├── Alexandra Daddario/
│   ├── Alia Bhatt/
│   ├── Amitabh Bachchan/
│   ├── Andy Samberg/
│   ├── Anushka Sharma/
│   ├── Billie Eilish/
│   ├── Brad Pitt/
│   ├── Camila Cabello/
│   ├── Charlize Theron/
│   ├── Claire Holt/
│   ├── Courtney Cox/
│   └── Dwayne Johnson/
│
├── attendance_captures/
│   ├── captured attendance face images
│
├── embeddings_out/
├── extracted_faces_from_group/
├── qdrant_storage/
├── attendance_log.csv
├── 4people.jpeg
├── 4people2.jpeg
├── 5people.jpeg
│
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/1fatmayz/capstone.git
```

Then open the project folder:

```bash
cd Attendance-Face-Detection
```

If your folder is inside another `capstone` folder, use:

```bash
cd "C:\Users\ffatm\OneDrive\Desktop\capstone\Attendance-Face-Detection"
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

---

## Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

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

Qdrant must be running before enrollment or recognition.

Run Qdrant locally using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Verify that Qdrant is running:

```text
http://localhost:6333/dashboard
```

The default Qdrant collection used in this project is:

```text
tryface
```

---

# Preparing the Dataset

Create a folder called:

```text
my_dataset/
```

Inside it, create one folder for each enrolled person.

Example:

```text
my_dataset/
│
├── Ahmed/
│   ├── image1.jpg
│   ├── image2.jpg
│
├── Fatma/
│   ├── image1.jpg
│   ├── image2.jpg
│
├── Shaima/
│   ├── image1.jpg
│   ├── image2.jpg
```

The preprocessing script will automatically create:

```text
clean_faces_4_white/
```

This folder contains the cleaned and aligned face images used for enrollment and recognition.

---

# Step 1 – Face Extraction and Preprocessing

Run:

```bash
python extract_clean_faces.py
```

This script:

* Reads images from `my_dataset/`
* Detects faces using MTCNN
* Aligns faces using facial landmarks
* Applies white background processing
* Resizes faces to 160×160 pixels
* Saves processed faces to `clean_faces_4_white/`

---

# Step 2 – Enroll Students into Qdrant

Run:

```bash
python savetoFaceDB.py
```

This script:

* Loads processed facial images from `clean_faces_4_white/`
* Generates FaceNet embeddings
* Creates one centroid embedding per person
* Stores the centroid vectors in Qdrant
* Saves metadata such as student name, image path, and number of images used

---

# Step 3 – Inspect Stored Qdrant Data

Run:

```bash
python check_qdrant_dataset.py
```

This script displays:

* Collection information
* Total stored records
* Unique enrolled names
* Payload keys
* Sample stored payloads

---

# Step 4 – Test Recognition Queries

Run:

```bash
python queryFaceDB.py
```

This script:

* Converts query images into embeddings
* Searches Qdrant
* Returns Top-K matches
* Applies threshold and gap logic
* Prints whether the face is known or unknown

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

* Detects multiple faces in one image
* Extracts and preprocesses each detected face
* Generates embeddings
* Searches Qdrant
* Labels known and unknown faces
* Saves cropped faces into `extracted_faces_from_group/`
* Saves attendance captures into `attendance_captures/`

---

# Step 6 – Evaluate System Performance

Run:

```bash
python evaluate_metrics.py
```

This script calculates:

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

* Generates embeddings from cleaned faces
* Builds positive and negative face pairs
* Tests multiple threshold values
* Helps find the best recognition threshold

---

# Launching the Web Application

Before launching the web application, make sure:

1. Qdrant is running.
2. The virtual environment is activated.
3. You are inside the correct project folder.

Run:

```bash
streamlit run app.py
```

The application will open on the computer at:

```text
http://localhost:8501
```

If PowerShell shows this error:

```text
Error: Invalid value: File does not exist: app.py
```

it means you are not inside the correct folder. Fix it by running:

```bash
cd "C:\Users\ffatm\OneDrive\Desktop\capstone\Attendance-Face-Detection"
streamlit run app.py
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
* Real-time face recognition
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

## Enroll Student

Allows:

* Adding new students
* Uploading or capturing student images
* Saving new face embeddings into Qdrant

## Settings

Allows:

* Threshold configuration
* Top-K configuration
* Gap configuration
* Qdrant connection settings

---

# Mobile Access

To access the Streamlit app on another device, use the ngrok public link.

First, run the Streamlit app:

```bash
streamlit run app.py
```

The Streamlit terminal should show:

```text
Local URL: http://localhost:8501
Network URL: http://172.22.1.2:8501
```

Then open another PowerShell window and run:

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

Do not use `localhost` on your phone because `localhost` only works on the computer running the app.

---

# Important Running Notes

* Keep the Qdrant terminal running.
* Keep the Streamlit terminal running.
* Keep the ngrok terminal running if using mobile access.
* Run `app.py` only from the correct project folder.
* If recognition does not work, check that Qdrant contains enrolled students.
* If no faces are detected, check image quality, lighting, and face angle.
* If the app cannot show matched images, check that the saved `image_path` exists on the computer.

---

# Output Folders

## `clean_faces_4_white/`

Contains cleaned, aligned, and resized face images used for enrollment.

## `attendance_captures/`

Contains face images captured during attendance recognition.

Example saved files:

```text
20260524_204736_Fatma_face1.jpg
20260601_143339_Ahmed_face1.jpg
20260601_143503_Fatma_face1.jpg
20260601_143503_Randa_face2.jpg
20260603_130816_Randa_face1.jpg
```

## `extracted_faces_from_group/`

Contains face crops extracted from group recognition images.

## `embeddings_out/`

Contains embedding analysis outputs generated by threshold training.

## `qdrant_storage/`

Contains local Qdrant-related storage files if Qdrant is configured locally.

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

* Using facial embeddings instead of raw image comparison
* Storing vector representations in Qdrant
* Performing recognition locally
* Reducing exposure of raw biometric data
* Supporting controlled enrollment and recognition

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
