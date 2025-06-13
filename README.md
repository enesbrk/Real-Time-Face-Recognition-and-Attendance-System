# Face Recognition Attendance System

This project implements an **AI-based facial recognition attendance system** designed to automate student attendance in university classrooms. It leverages face recognition technology to identify students and records their attendance in real-time using Firebase and Google Cloud Storage.

## 🚀 Features

- 📸 Face recognition-based automatic attendance
- 🧑‍🎓 Student registration by faculty, department, and class
- ☁️ Dataset stored in Google Cloud Storage (GCS) with hierarchical folder structure
- 🔄 Auto-update for new registered faces to avoid duplicate processing
- 🖼️ User-friendly GUI built with Tkinter
- 🔥 Real-time attendance logging via Firebase Realtime Database

## 📁 Project Structure

project/

 - register_mode.py           # GUI for student registration
 - register_logic.py          # Handles photo capture and GCS upload
 - attendance_mode.py         # Face recognition and attendance marking
 - known_faces.pkl            # Pickled face embeddings
 - firebase_config.json       # Firebase configuration file
 - gcs_utils.py               # GCS upload and folder management
 - gui_app.py                 # Main GUI application launcher
 - dataset/                   # Optional local dataset (for debugging)

## 🧰 Technologies Used

- Python (face_recognition, OpenCV, Tkinter)
- Google Cloud Storage (GCS)
- Firebase Realtime Database
- Pickle (for serializing known faces)

## 🏗️ GCS Folder Structure

Images are organized hierarchically:

Faculty/
    Department/
        Class/
            Student_Name_Surname/
                1.jpg
                2.jpg    


## ⚙️ Installation

1. Clone the repository:
   git clone https://github.com/yourusername/face-recognition-attendance.git
   cd face-recognition-attendance

2. Install dependencies:
   pip install -r requirements.txt

3. Add your firebase_config.json file to the root directory.

4. Ensure GCS credentials and bucket settings are correctly configured in gcs_utils.py.

## ▶️ Usage

### Register Mode

python register_mode.py

- Launches the registration GUI.
- Enter student info (name, faculty, department, class).
- Press s to capture and upload images to GCS.

### Attendance Mode

python attendance_mode.py

- Opens camera.
- Recognizes faces and records attendance in Firebase.
- Only registered faces are recognized; new ones are ignored.

## 📌 Notes

- The system skips re-processing known faces by checking known_faces.pkl.
- Ensure Firebase and GCS integration is properly configured before deployment.

