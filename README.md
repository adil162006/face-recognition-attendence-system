# 🧠 Face Recognition Attendance System

This project is a **smart attendance system** that uses **face recognition** to automatically detect and log students or employees' attendance via webcam, storing their name, time, and date in a CSV file.

---

## 📸 Features

- Real-time face recognition using webcam
- Automatic attendance marking in `attendance.csv`
- Avoids duplicate entries during the session
- Scalable: just add more known face images
- Lightweight and runs on any basic PC with a webcam

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| `face_recognition` | Detect and recognize faces |
| `opencv-python` | Webcam access and drawing boxes |
| `numpy` | Handle facial encodings |
| `datetime` | Timestamp for attendance |
| `csv` | Save data into a spreadsheet format |

---

## 📂 Project Structure

