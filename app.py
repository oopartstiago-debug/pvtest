import os
import pathlib

os.makedirs("/tmp/.tabpfn", exist_ok=True)
PATCHED_PATH = pathlib.Path("/tmp/.tabpfn/token")

import streamlit as st
st.set_page_config(page_title="Debug")
st.title("Debug")

token = st.secrets["TABPFN_TOKEN"]

# 모든 관련 클래스 경로 일괄 패치
from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.client import ServiceClient
import tabpfn_client
from tabpfn_client import constants

# 패치 가능한 모든 경로 속성 덮어쓰기
UserAuthenticationClient.CACHED_TOKEN_FILE = PATCHED_PATH

for cls in [ServiceClient, UserAuthenticationClient]:
    for attr in dir(cls):
        try:
            val = getattr(cls, attr)
            if isinstance(val, pathlib.Path) and ".tabpfn" in str(val):
                setattr(cls, attr, PATCHED_PATH.parent / val.name)
                st.write(f"Patched {cls.__name__}.{attr} → {getattr(cls, attr)}")
        except Exception:
            pass

# constants 모듈 경로도 패치
constants.CACHE_DIR = PATCHED_PATH.parent

tabpfn_client.set_access_token(token)
st.success("set_access_token() succeeded!")

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
    import traceback
    st.error(f"fit failed: {e}")
    st.code(traceback.format_exc())
