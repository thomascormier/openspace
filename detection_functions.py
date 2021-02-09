#import streamlit as st
import cv2
import numpy as np
#import pandas as pd
#import os
import face_recognition
from csv_functions import create_csv, update_csv
from csv_trombinoscope import load_known_data, init_data
#from mail import send_mail, update_csv_mail
#from smtplib import SMTPRecipientsRefused
#import SessionState
#from streamlit import caching

# constants trombinoscope
COLOR_DARK = (0, 0, 153)
known_face_names, known_face_encodings = load_known_data()

# constants code initial
confThreshold = 0.5
nmsThreshold = 0.3
nb_pers = 0


classesFile = 'coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

def capture_face(video_capture):
    # got 3 frames to auto adjust webcam light
    video_capture.read()

    while True:
        ret, frame = video_capture.read()
        # FRAME_WINDOW.image(frame[:, :, ::-1])
        image_placeholder.image(frame[:, :, ::-1])
        # face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if len(face_locations) > 0:
            video_capture.release()
            return frame


##################################### functions trombinoscope


def recognize_frame(frame):
    # convert COLOR_BGR2RGB
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_DARK, 2)

        # Draw a label with a name below the face
        cv2.putText(frame, name, (left + 10, bottom + 15),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

def byte_to_array(image_in_byte):
    return cv2.imdecode(
        np.frombuffer(image_in_byte.read(), np.uint8),
        cv2.IMREAD_COLOR
    )


def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

# convert face distance to similirity likelyhood


def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))


##################################### functions


def findObjects(outputs, img, threshold):
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
                w = int(det[2] * wT)
                h = int(det[3] * hT)
                x = int((det[0] * wT) - w / 2)
                y = int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    global nb_pers
    nb_pers = 0
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255))
        if classNames[classIds[i]].upper() == "PERSON":
            nb_pers += 1
    update_csv(nb_pers, threshold)
    return nb_pers

