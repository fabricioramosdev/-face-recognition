import cv2
import numpy as np
import os

################# KNN CODE ######################
def distance(V1, V2):
    return np.sqrt(((V1-V2)**2).sum())

def Knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        ix =  train[i, :-1]
        iy = train[i, -1]

        d = distance(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])

    return output[0][index]

# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')

skip = 0
dataset_path = './data/'

face_data = []
labels = []

class_id = 0
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        print('Loaded '+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1,1))
# print(face_dataset.shape)
# print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
# print(trainset.shape)

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h),  (255,86, 86), 2)
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = Knn(trainset, face_section.flatten())
        predict_name = names[int(out)]
        cv2.putText(frame, predict_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,86, 86), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h),  (255,86, 86), 2)

    cv2.imshow('Faces', frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
