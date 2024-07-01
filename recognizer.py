import cv2
import face_recognition
import os
from datetime import datetime


log_file = open('logs.txt', 'a')
video_capture = cv2.VideoCapture(0)


faces = {}

for filename in os.listdir('storage'):
    if filename.endswith('.txt'):
        with open(f'storage/{filename}', 'r') as f:
            saved_encodings = [[float(x) for x in line.split()] for line in f]
            name = os.path.splitext(filename)[0]
            
            faces[name] = saved_encodings


while True:
    ret, frame = video_capture.read()

    recognized = {}

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        best_match_name = ""
        best_match_coeff = 0
        
        for name in faces:
            matches = face_recognition.compare_faces(faces[name], face_encoding)
            face_distances = face_recognition.face_distance(faces[name], face_encoding)
            match_coeff = 1 - face_distances.mean()

            if match_coeff > best_match_coeff:
                best_match_coeff = match_coeff
                best_match_name = name

        if best_match_coeff >= 0.5:
            log_file.write(f"{best_match_name} - {datetime.now()}\n")
            recognized[best_match_name] = (top, right, bottom, left)
        else:
            log_file.write(f"New face - {datetime.now()}\n")
            recognized['new'] = (top, right, bottom, left)



    for name in recognized:
        top, right, bottom, left = recognized[name]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
