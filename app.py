import os
import sys

# ── 반드시 tabpfn_client import 전에 설정 ──
os.environ["TABPFN_CONFIG_DIR"] = "/tmp/.tabpfn"
os.makedirs("/tmp/.tabpfn", exist_ok=True)

import streamlit as st
st.set_page_config(page_title="Debug")
st.title("Debug")

# secrets에서 토큰 직접 읽기
try:
    token = st.secrets["TABPFN_TOKEN"]
    st.success(f"Token OK: length={len(token)}")
except Exception as e:
    st.error(f"secrets read failed: {e}")
    st.stop()

import tabpfn_client
st.write(f"tabpfn_client config dir: {os.environ.get('TABPFN_CONFIG_DIR')}")

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
