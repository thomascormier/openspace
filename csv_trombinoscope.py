import streamlit as st
import os
import numpy as np
import pandas as pd

PATH_DATA = 'DB.csv'
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


def init_data(data_path=PATH_DATA):
    if os.path.isfile(data_path):
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
