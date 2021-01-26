import streamlit as st
import pandas as pd
from random import *
import csv
import cv2


def display_image():
    if number_of_people == 0:
        st.image("img/ippon_openspace0.jpeg", use_column_width=True)
    elif number_of_people == 1:
        st.image("img/ippon_openspace1.jpeg", use_column_width=True)
    elif number_of_people == 2:
        st.image("img/ippon_openspace2.jpeg", use_column_width=True)
    elif number_of_people == 3:
        st.image("img/ippon_openspace3.jpeg", use_column_width=True)
    elif number_of_people > 3:
        st.image("img/ippon_openspace4.jpeg", use_column_width=True)


st.write("# Room dashboard")

st.write("This demo view show you what the final view is suppose to look like.")

unuse_threshlod = st.slider("The user can set a Threshold. For the demo, it is set to 3, you can play with the slider but it's not connected to the image :", 1, 5)
setted_threshlod = 3
st.sidebar.write("Demo threshlod : ", setted_threshlod)


number_of_people = st.sidebar.slider("Here you can change the number of people that the model recognizes :", 0, 4)
st.sidebar.write("You can see that the color of the number of person on the page changes. Also, the color of the rectangles around people changes depending one how much people are in the room")

filling = round(number_of_people/setted_threshlod, 2)*100


with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

if number_of_people < setted_threshlod:
    text = "## <div>Occupation : <span class='highlight green'>" + str(number_of_people) + "</span> / " + str(setted_threshlod) + "</div>"
    st.write(text, unsafe_allow_html=True)
elif filling == 100:
    text = "## <div>Occupation : <span class='highlight orange'>" + str(number_of_people) + "</span> / " + str(setted_threshlod) + "</div>"
    st.write(text, unsafe_allow_html=True)
elif filling > 100:
    text = "## <div>Occupation : <span class='highlight red'>" + str(number_of_people) + "</span> / " + str(setted_threshlod) + "</div>"
    st.write(text, unsafe_allow_html=True)


display_image()


