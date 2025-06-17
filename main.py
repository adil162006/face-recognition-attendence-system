import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import csv

# Load known faces
known_face_encodings = []
known_face_names = []

folder_path = 'known_faces'

for file_name in os.listdir(folder_path):
    if file_name.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(folder_path, file_name)
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(file_name)[0])  # Remove extension

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Used to keep track of who has been marked
marked_names = set()

def mark_attendance(name):
    if name not in marked_names:
        now = datetime.now()
        time_string = now.strftime('%H:%M:%S')
        date_string = now.strftime('%Y-%m-%d')
        with open('attendance.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, time_string, date_string])
        marked_names.add(name)
        print(f"Marked {name} at {time_string} on {date_string}")

print("Press 'q' to quit.")
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)

            # Draw rectangle
            top, right, bottom, left = [v*4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Face Recognition Attendance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
