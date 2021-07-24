import numpy as np
import face_recognition
import cv2

video = cv2.VideoCapture(0)
classificadorFace = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')

while True:
    conectado, frame = video.read()
    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(frameCinza, 1.3, 5)
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255,86, 86), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()