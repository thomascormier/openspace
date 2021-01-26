import streamlit as st
import pandas as pd
from random import *
import csv
import cv2


def generate_data():
    f = open('data_for_graph.csv', 'w')
    with f:

        fnames = ['number_of_people', 'threshold']
        writer = csv.DictWriter(f, fieldnames=fnames)

        writer.writeheader()

        cpt = 0
        for i in range(50):
            n = randint(-1, 1)
            if cpt + n <= 0:
                cpt = 0
            elif cpt + n == 6:
                cpt += -1
            else:
                cpt = cpt + n
            writer.writerow({'number_of_people': cpt, 'threshold': seuil})
    print('New csv generated')

seuil = 3

st.write("# Demo display graph from csv")
st.write("Threshold is set to : ", seuil)

st.write("If you want to display a new random dataset click on the button below :")
if st.button("Generate new dataset"):
    generate_data()

df = pd.read_csv("data_for_graph.csv")
st.line_chart(df)