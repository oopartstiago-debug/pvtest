import os
import inspect
import pathlib

os.makedirs("/tmp/.tabpfn", exist_ok=True)

import streamlit as st
st.set_page_config(page_title="Debug")
st.title("Debug")

token = st.secrets["TABPFN_TOKEN"]

from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.constants import CACHE_DIR

st.subheader("CACHE_DIR in constants:")
st.write(str(CACHE_DIR))

st.subheader("UserAuthenticationClient.set_token source:")
st.code(inspect.getsource(UserAuthenticationClient.set_token))

