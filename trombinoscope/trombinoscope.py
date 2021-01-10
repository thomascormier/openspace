import streamlit as st
import os
import numpy as np
import pandas as pd
import face_recognition
import cv2

# CONSTANTS
PATH_DATA = 'data/DB.csv'
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['name', 'description']
COLS_ENCODE = [f'v{i}' for i in range(128)]


@st.cache
def load_known_data():
    DB = pd.read_csv(PATH_DATA)
    return (
        DB['name'].values,
        DB[COLS_ENCODE].values
        )


def capture_face(video_capture):
    # got 3 frames to auto adjust webcam light
    video_capture.read()

    while True:
        ret, frame = video_capture.read()
        FRAME_WINDOW.image(frame[:, :, ::-1])
        # face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if len(face_locations) > 0:
            video_capture.release()
            return frame


def recognize_frame(frame):
    # convert COLOR_BGR2RGB
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Draw a box around the face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        name = known_face_names[best_match_index]
        similarity = face_distance_to_conf(face_distances[best_match_index], 0.5)
        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_DARK, 2)
        cv2.putText(frame, name, (left + 10, bottom + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        return name, similarity, frame[:, :, ::-1]


def init_data(data_path=PATH_DATA):
    if os.path.isfile(data_path):
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)


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


def main():
    activities = ["Upload Photo", "Trombinoscope", "About"]

    choice = st.sidebar.selectbox("Select Activty", activities)
########################################################################
    if choice == 'About':
        st.subheader("About Face Detection App")
        st.markdown("Built with Streamlit by Alexandra, Thomas and Andrei")
        st.text("Projet Fin Etudes OpenSpace")
#########################################################################
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
#####################################################################################
    elif choice == 'Trombinoscope':
        st.subheader("Trombinoscope")
        if st.button('Start', key="start camera"):
            alert_placeholder = st.empty()
            while True:
                video_capture = cv2.VideoCapture(0)
                frame = capture_face(video_capture)
                name, similarity, frame = recognize_frame(frame)
                FRAME_WINDOW.image(frame)
                alert_placeholder.write("")
                if similarity > 0.75:
                    # label = f"**{name}**: *{similarity:.2%} likely*"
                    # st.markdown(label)
                    alert_placeholder.text(f"{name}: {similarity:.2%} likely")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == '__main__':
    # top text
    """People Detection App"""

    st.title("Face Detection App")
    st.text("Build with Streamlit and OpenCV")
    FRAME_WINDOW = st.image([])
    known_face_names, known_face_encodings = load_known_data()
    main()
