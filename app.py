import os
import pathlib

os.makedirs("/tmp/.tabpfn", exist_ok=True)

import streamlit as st
st.set_page_config(page_title="Debug")
st.title("Debug")

token = st.secrets["TABPFN_TOKEN"]
st.success(f"Token OK: length={len(token)}")

# tabpfn_client config 경로를 /tmp로 강제 패치
import tabpfn_client
from tabpfn_client import config as tabpfn_config

tabpfn_config.CACHE_DIR = pathlib.Path("/tmp/.tabpfn")
st.write(f"CACHE_DIR patched to: {tabpfn_config.CACHE_DIR}")

try:
    tabpfn_client.set_access_token(token)
    st.success("set_access_token() succeeded")
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
