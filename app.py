import os
import streamlit as st

st.set_page_config(page_title="Debug")
st.title("Debug")

# secrets에서 직접 읽기 (환경변수 우회)
try:
    token = st.secrets["TABPFN_TOKEN"]
    st.success(f"Token from secrets: length={len(token)}")
except Exception as e:
    st.error(f"secrets read failed: {e}")
    st.stop()

import tabpfn_client

# 토큰 저장 경로를 쓰기 가능한 곳으로 변경
token_dir = "/tmp/.tabpfn"
os.makedirs(token_dir, exist_ok=True)
os.environ["TABPFN_CONFIG_DIR"] = token_dir

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
