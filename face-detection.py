import cv2
import face_recognition

name = input("Enter the name of the person: ")

video_capture = cv2.VideoCapture(0)


face_encodings = []


counter = 0

while True:
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):

        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) == 1:

            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
            face_encodings.append(face_encoding)
            counter += 1
            print(f"Captured image {counter} for {name}.")

        if counter >= 10:
            break


if face_encodings:
    with open(f'storage/{name}.txt', 'w') as f:
        for encoding in face_encodings:
            f.write(' '.join(map(str, encoding.tolist())) + '\n')


video_capture.release()
cv2.destroyAllWindows()