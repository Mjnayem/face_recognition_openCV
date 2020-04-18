import cv2
import numpy as np

face_classifire = cv2.CascadeClassifier('/var/www/python/opencv/haarcascade_frontalface_default.xml')


def face_extractor(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifire.detectMultiScale(grey, 1.5, 5)
    if faces is ():
        return None

    for(x, y, w, h) in faces:
        cropped_face = img[y:y+h, h:x+w]

    return cropped_face


cap = cv2.VideoCapture('http://192.168.0.102:4747/video')
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame),(200, 200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_name_path = '/var/www/python/opencv/demo_4/images/'+str(count)+'.jpg'
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
    else:
        print('Face not found !')
        pass

    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print('Sample collection completed!!!')
