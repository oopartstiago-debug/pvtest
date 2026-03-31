import os
import inspect
import pathlib

os.makedirs("/tmp/.tabpfn", exist_ok=True)

import streamlit as st
st.set_page_config(page_title="Debug")
st.title("Debug")

token = st.secrets["TABPFN_TOKEN"]

import tabpfn_client
from tabpfn_client import config as tabpfn_config

# set_access_token 소스코드 출력
st.subheader("set_access_token source:")
st.code(inspect.getsource(tabpfn_client.set_access_token))

# config 전체 소스도 확인
st.subheader("config.py source:")
st.code(inspect.getsource(tabpfn_config))
