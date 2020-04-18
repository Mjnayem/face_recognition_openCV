import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('/var/www/python/opencv/haarcascades/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('/var/www/python/opencv/haarcascades/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

labels = {}

with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)
    labels = {v:k for k, v in labels.items()}

cap = cv2.VideoCapture('http://192.168.0.102:4747/video')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for(x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 :
            name = labels[id_]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 180, 150), 2)
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        # detect eye
        eyes = eyes_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (50, 30, 40), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()