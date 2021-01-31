import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import face_recognition
from csv_functions import create_csv, update_csv
from csv_trombinoscope import load_known_data, init_data
from mail import send_mail, update_csv_mail
from smtplib import SMTPRecipientsRefused

# constants trombinoscope
PATH_DATA = 'DB.csv'
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['name', 'description']
COLS_ENCODE = [f'v{i}' for i in range(128)]
known_face_names, known_face_encodings = load_known_data()

# constants code initial
cap = cv2.VideoCapture(0)
whT = 608
confThreshold = 0.5
nmsThreshold = 0.3
nb_pers = 0


classesFile = 'coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
##################################### functions face store


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
    update_csv(nb_pers)
    return nb_pers


def main():
########################################### main general
    """People Detection App"""

    st.title("Open space people counter")
    st.text("Build with Streamlit and OpenCV")

    activities = ["Dashboard", "Upload Photo","Add Supervisors", "About"]

    choice = st.sidebar.selectbox("Select Activty", activities)


########################################### main About
    if choice == 'About':
        st.subheader("About Open Space People Counter App")
        st.markdown("Built in Streamlit by Alexandra, Thomas and Andrei")
        st.text("Projet Fin Etudes OpenSpace")
        st.text("Projet proposé par Ippon Technologies")


########################################### main About
    if choice == 'Add Supervisors':

        person_name = st.text_input('Name:', '')
        person_email = st.text_input('Email:', '')
        if st.button('Add'):
            update_csv_mail(person_email, person_name)
        if st.button('Reset'):
            os.remove('email.csv')
            name = 'email.csv'
            field1 = 'EMAIL'
            field2 = 'NAME'
            create_csv(name, field1, field2)

########################################### main face capture
    elif choice == 'Upload Photo':
        # disable warning signs:
        st.set_option("deprecation.showfileUploaderEncoding", False)

        st.subheader("Upload Photo")

        # displays a file uploader widget and return to BytesIO
        image_byte = st.file_uploader(
            label="Select a picture containing faces:", type=['jpg', 'png']
        )
        # detect faces in the loaded image
        max_faces = 0
        rois = []  # region of interests (arrays of face areas)
        if image_byte is not None:
            image_array = byte_to_array(image_byte)
            face_locations = face_recognition.face_locations(image_array)
            for idx, (top, right, bottom, left) in enumerate(face_locations):
                # save face region of interest to list
                rois.append(image_array[top:bottom, left:right].copy())

                # Draw a box around the face and label it
                cv2.rectangle(image_array, (left, top),
                              (right, bottom), COLOR_DARK, 2)
                cv2.rectangle(
                    image_array, (left, bottom + 35),
                    (right, bottom), COLOR_DARK, cv2.FILLED
                )
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    image_array, f"#{idx}", (left + 5, bottom + 25),
                    font, .55, COLOR_WHITE, 1
                )

            st.image(BGR_to_RGB(image_array), width=720)
            max_faces = len(face_locations)

        if max_faces > 0:
            # select interested face in picture
            face_idx = st.selectbox("Select face#", range(max_faces))
            roi = rois[face_idx]
            st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))

            # initial database for known faces
            DB = init_data()
            face_encodings = DB[COLS_ENCODE].values
            dataframe = DB[COLS_INFO]
            try:
                # compare roi to known faces, show distances and similarities
                face_to_compare = face_recognition.face_encodings(roi)[0]
                dataframe['distance'] = face_recognition.face_distance(
                    face_encodings, face_to_compare
                )
                dataframe['similarity'] = dataframe.distance.apply(
                    lambda distance: f"{face_distance_to_conf(distance):0.2%}"
                )
                st.dataframe(
                    dataframe.sort_values("distance").iloc[:5]
                        .set_index('name')
                )

                # add roi to known database
                if st.checkbox('Add it to known faces'):
                    face_name = st.text_input('Name:', '')
                    face_des = st.text_input('Desciption:', '')
                    if st.button('Add'):
                        encoding = face_to_compare.tolist()
                        DB.loc[len(DB)] = [face_name, face_des] + encoding
                        DB.to_csv(PATH_DATA, index=False)
            except IndexError:
                st.subheader("This image is not relatable to our dataset")
        else:
            st.write('No human faces detected.')


########################################### main trombinoscope + detection
    elif choice == 'Dashboard':

        nb_person_threshold = st.slider("Define threshold : ", 2, 5)

        alert_placeholder = st.empty()
        image_placeholder = st.empty()
        people_placeholder = st.empty()
        graph_placeholder = st.empty()
        email_placeholder = st.empty()

        name = 'number_of_people.csv'
        field1 = 'number_of_people'
        field2 ='nb_person_threshold'
        create_csv(name,field1,field2)


        while True:
            success, img = cap.read()

            width = 640
            height = 480
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
            net.setInput(blob)

            layerNames = net.getLayerNames()
            outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            outputs = net.forward(outputNames)

            nb_pers = findObjects(outputs, img)

            try:
                name, similarity, img = recognize_frame(img)
            except TypeError:
                variable = 0


            people_placeholder.write(f"There are {nb_pers} people here!")

            b, g, r = cv2.split(img)  # get b, g, r
            rgb_img1 = cv2.merge([r, g, b])  # switch it to r, g, b

            image_placeholder.image(rgb_img1)

            alert_placeholder.write("")
            if nb_pers >= nb_person_threshold:
                alert_placeholder.error(
                    f"ALERT! There should be less than {nb_person_threshold} people and there are {nb_pers} people")
                try:
                    send_mail()
                except SMTPRecipientsRefused:
                    email_placeholder.error(f"ATTENTION! One or more email adresses in the database is not valid, please reset and try once more!")

            df = pd.read_csv("number_of_people.csv")
            graph_placeholder.line_chart(df)


if __name__ == '__main__':
    main()