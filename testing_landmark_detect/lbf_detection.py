import cv2
import numpy as np

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

facemark = cv2.face.createFacemarkLBF()
facemark.loadModel('lbfmodel.yaml')

img = cv2.imread('image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

if len(faces) > 0:
    _, landmarks = facemark.fit(gray, faces)

    for landmark in landmarks:
        for x, y in landmark[0]:
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
else:
    print("Không phát hiện được khuôn mặt.")

cv2.imshow('Landmarks', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
