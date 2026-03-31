import os
import sys

os.makedirs("/tmp/.tabpfn", exist_ok=True)

import streamlit as st
st.set_page_config(page_title="Debug")
st.title("Debug")

token = st.secrets["TABPFN_TOKEN"]
st.success(f"Token OK: length={len(token)}")

# tabpfn_client 내부 config 경로를 import 후 직접 패치
import tabpfn_client
from tabpfn_client import config as tabpfn_config

# config 모듈이 어떤 경로를 쓰는지 확인
st.write("config module attributes:")
st.write([x for x in dir(tabpfn_config) if not x.startswith("__")])

# 경로 관련 속성 값 출력
for attr in dir(tabpfn_config):
    if not attr.startswith("__"):
        val = getattr(tabpfn_config, attr)
        if isinstance(val, (str, os.PathLike)):
            st.write(f"{attr} = {val}")
