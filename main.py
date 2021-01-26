import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from csv_functions import create_csv, update_csv

cap = cv2.VideoCapture(0)
whT = 608
confThreshold = 0.5
nmsThreshold = 0.3
nb_pers = 0

classesFile = 'yolo_files/coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classId == 0:
                w = int(det[2]*wT)
                h = int(det[3]*hT)
                x = int((det[0]*wT) - w/2)
                y = int((det[1]*hT) - h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    global nb_pers
    nb_pers = 0
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w ,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 255))
        if classNames[classIds[i]].upper() == "PERSON":
            nb_pers += 1
    update_csv(nb_pers)
    return nb_pers


def main():
    st.title("Open space people counter")

    nb_person_threshold = st.slider("Define threshold : ", 2, 5)

    alert_placeholder = st.empty()
    image_placeholder = st.empty()
    people_placeholder = st.empty()
    graph_placeholder = st.empty()

    create_csv()
    os.remove('number_of_people.csv')
    
    while True:
        success, img = cap.read()

        width = 640
        height = 480
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(outputNames)

        nb_pers = findObjects(outputs, img)

        people_placeholder.write(f"There are {nb_pers} people here!")

        image_placeholder.image(img, channels="BGR")

        alert_placeholder.write("")
        if nb_pers >= nb_person_threshold:
            alert_placeholder.error(f"ALERT! There should be less than {nb_person_threshold} people and there are {nb_pers} people")

        df = pd.read_csv("number_of_people.csv")
        graph_placeholder.line_chart(df)

        


if __name__ == '__main__':
    main()
