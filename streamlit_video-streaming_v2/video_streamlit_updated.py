# Core Pkgs
import streamlit as st
import cv2
import numpy as np
import time
import imutils
import argparse
#import pickle


def detect(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    bounding_box_cordinates, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)

    person = 1
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1

    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons : {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    return frame


def main():
    """People Detection App"""

    st.title("Face Detection App")
    st.text("Build with Streamlit and OpenCV")

    activities = ["Load Video",  "Camera", "Person Recon Camera", "About"]
    # "Load Video+Algo", this goes here   ^
    choice = st.sidebar.selectbox("Select Activty", activities)

    if choice == 'Load Video':
        st.subheader("Load Video")

        video_file = st.file_uploader("Upload Video", type=['mp4'])

        if video_file is not None:
            st.text("Original Video")
            video_bytes = video_file.read()
            # st.write(type(our_image))
            st.video(video_bytes)

    elif choice == 'About':
        st.subheader("About Face Detection App")
        st.markdown("Built with Streamlit by Alexandra, Thomas and Andrei")
        st.text("Projet Fin Etudes OpenSpace")

    elif choice == 'Camera':
        st.subheader("Camera")
        if st.button('Start', key="start camera"):

            video = cv2.VideoCapture(0)
            video.set(cv2.CAP_PROP_FPS, 25)

            image_placeholder = st.empty()

            while True:
                success, image = video.read()
                if not success:
                    break
                image_placeholder.image(image, channels="BGR")
                time.sleep(0.01)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    elif choice == 'Person Recon Camera':
        st.subheader("Person Recon Camera")

        out = cv2.VideoWriter(
             'output.avi',
             cv2.VideoWriter_fourcc(*'MJPG'),
             15.,
             (640, 480))
        if st.button('Start', key="start camera recon"):
            video = cv2.VideoCapture(0)
            video.set(cv2.CAP_PROP_FPS, 25)

            image_placeholder = st.empty()
            while True:
                check, frame = video.read()
                frame = detect(frame)
                out.write(frame.astype('uint8'))
                image_placeholder.image(frame, channels="BGR")
                time.sleep(0.01)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # cannot solve the algorithm on the doc because it needs a numpy.ndarray and i get bytes from st.file_uploader.
    # but i dont want to erase the code either cuz i lost a ton of time trying to make the damn thing work
    # elif choice == 'Load Video+Algo':
    #     st.subheader("Load Video+Algo")
    #
    #     video_file = st.file_uploader("Upload Video", type=['mp4'])
    #     if video_file is not None:
    #         st.text("Original Video")
    #         video_bytes = video_file.read()
    #         # st.write(type(our_image))
    #         st.video(video_bytes)
    #     if st.button("Start Check"):
    #         while True:
    #             frame = video_file.read()
    #             frame = detect(frame)
    #             st.video(frame)


if __name__ == '__main__':
    main()
