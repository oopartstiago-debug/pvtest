import os
import inspect
import pathlib

os.makedirs("/tmp/.tabpfn", exist_ok=True)

import streamlit as st
st.set_page_config(page_title="Debug")
st.title("Debug")

token = st.secrets["TABPFN_TOKEN"]

# CACHED_TOKEN_FILE 경로를 /tmp로 패치 (import 직후)
from tabpfn_client.service_wrapper import UserAuthenticationClient
UserAuthenticationClient.CACHED_TOKEN_FILE = pathlib.Path("/tmp/.tabpfn/token")
st.write(f"CACHED_TOKEN_FILE patched to: {UserAuthenticationClient.CACHED_TOKEN_FILE}")

import tabpfn_client

try:
    tabpfn_client.set_access_token(token)
    st.success("set_access_token() succeeded!")
except Exception as e:
    st.error(f"set_access_token failed: {e}")
    st.stop()

try:
    from tabpfn_client import TabPFNRegressor
    import numpy as np
    import pandas as pd

    X = pd.DataFrame({"a": [1,2,3], "b": [4,5,6]})
    y = np.array([1.0, 2.0, 3.0])
    m = TabPFNRegressor()
    m.fit(X, y)
    st.success("TabPFNRegressor fit succeeded!")
except Exception as e:
    st.error(f"fit failed: {e}")
