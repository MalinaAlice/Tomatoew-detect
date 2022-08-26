from ast import While
import cv2
import numpy as np

thres = 0.45  # Threshold to detect object

cap = cv2.VideoCapture("two.mp4")
cap.set(3, 1280)
cap.set(4, 300)
cap.set(10, 70)


# classNames = ['rosie-coapta', 'rosie-necoapta']
classNames = []
classFile = 'obj.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'custom-yolov4-tiny-detector.cfg'
weightsPath = 'custom-yolov4-tiny-detector_last.weights'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(380, 380)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Output', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
