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
import SessionState
from streamlit import caching

from detection_functions import capture_face
from detection_functions import recognize_frame
from detection_functions import byte_to_array
from detection_functions import BGR_to_RGB
from detection_functions import face_distance_to_conf
from detection_functions import findObjects

# constants trombinoscope
PATH_DATA = 'DB.csv'
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['name', 'description']
COLS_ENCODE = [f'v{i}' for i in range(128)]

# constants code initial
cap = cv2.VideoCapture(0)
whT = 608
nb_pers = 0

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def main():
########################################### main general
    """People Detection App"""
    activities = ["Add Supervisors", "Dashboard", "Upload Photo",  "About"]

    choice = st.sidebar.selectbox("Select Activity", activities)
    photo_placeholder=st.sidebar.empty()

########################################### main About
    if choice == 'About':
        st.sidebar.image('info.png')
        st.title("About Open Space People Counter App")
        st.markdown("Built in Streamlit by Alexandra, Thomas and Andrei")
        st.text("Projet Fin Etudes OpenSpace")
        st.text("Projet proposé par Ippon Technologies")
        st.write("Dans un premier temps, l’objectif du projet est d’analyser des vidéos prises dans différentes pièces (open space d’Ippon, magasin etc...), être capable de dénombrer les personnes présentes sur la vidéo et proposer un affichage de cette vidéo et de ce nombre sur un dashboard. Le nombre de personnes sera recalculé pendant toute la durée de la vidéo. Ainsi ce nombre sera mis à jour chaque seconde pour rester cohérent avec ce que l’on voit sur la vidéo. ")
        st.write("Afin d'être reconnu dans le flux vidéo en direct, une page Upload Photo a été conçue pour enregistrer les photos des personnes. Si la personne est déjà inscrite, la personne sur la photo que nous voulons ajouter sera identifiée au moment de la tentative d’enregistrement. Si les personnes présentes dans le flux vidéo n'ont pas été préalablement enregistrées, elles seront définies comme inconnues. Pour la fonction d'alerte par e-mail, l'utilisateur doit ajouter des adresses e-mail viables à l'aide de la page Add Supervisors. S'il n'y a pas d'adresse e-mail enregistrée, une alerte s'affiche sur les pages correspondant au tableau de bord. La même chose se produira lorsque le format d'une ou plusieurs adresses e-mail n'est pas approprié.")
        caching.clear_cache()

########################################### main About

    if choice == 'Add Supervisors':
        col1, col2, col3 = st.beta_columns(3);
        with col2:
            st.title("Add Supervisors")
            st.sidebar.image('supervisor.png')
            #col1, col2, col3= st.beta_columns(3);
            #with col2:
            person_name = st.text_input('Name:', '')
            person_email = st.text_input('Email:', '')
            if st.button('Add'):
                update_csv_mail(person_email, person_name)
        caching.clear_cache()

########################################### main face capture
    elif choice == 'Upload Photo':
        st.title("Upload Photo")
        st.sidebar.image('video.png')
        col1, col2,col3 = st.beta_columns((1,3,1));
        # disable warning signs:
        with col2:
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
                    var = 0
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
                        face_des = st.text_input('Description:', '')
                        if st.button('Add'):
                            encoding = face_to_compare.tolist()
                            DB.loc[len(DB)] = [face_name, face_des] + encoding
                            DB.to_csv(PATH_DATA, index=False)
                            var = 1
                except IndexError:
                    st.subheader("This image is not relatable to our dataset")
            else:
                st.write('No human faces detected.')


        caching.clear_cache()


########################################### main trombinoscope + detection

    elif choice == 'Dashboard':
        st.title("Dashboard")
        st.sidebar.image('dash.png')
        nb_person_threshold = st.slider("Define threshold : ", 1, 5)
        people_placeholder = st.empty()
        name = 'number_of_people.csv'
        field1 = 'number_of_people'
        field2 ='nb_person_threshold'
        create_csv(name, field1, field2)

        col1, col2 = st.beta_columns(2)
        with col1:
            image_placeholder = st.empty()
        with col2:
            alert_placeholder = st.empty()
            email_placeholder = st.empty()
        graph_placeholder = st.empty()
        while True:
            success, img = cap.read()
            width = 600
            height = 480
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
            net.setInput(blob)

            try:
                name, similarity, img = recognize_frame(img)
            except TypeError:
                variable = 0

            layerNames = net.getLayerNames()
            outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            outputs = net.forward(outputNames)

            nb_pers = findObjects(outputs, img, nb_person_threshold)

            people_placeholder.subheader(f"There are {nb_pers} people here!")

            b, g, r = cv2.split(img)  # get b, g, r
            rgb_img1 = cv2.merge([r, g, b])  # switch it to r, g, b

            image_placeholder.image(rgb_img1)

            alert_placeholder.write("")
            email_placeholder.write("")
            if nb_pers > nb_person_threshold:
                alert_placeholder.error(
                    f"ALERT! There should be less than {nb_person_threshold} people and there are {nb_pers} people")
                try:
                    send_mail()
                except SMTPRecipientsRefused:
                    email_placeholder.error(f"ATTENTION! One or more email adresses in the database is not valid, please reset and try once more!")
                except IndexError:
                    email_placeholder.error(f"ATTENTION! There are no email adresses inserted yet. Go insert at least one in the Add Supervisors panel and try again after!")

            df = pd.read_csv("number_of_people.csv")
            graph_placeholder.line_chart(df)
        caching.clear_cache()



if __name__ == '__main__':
    main()