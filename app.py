import os
import streamlit as st

st.set_page_config(page_title="Debug")
st.title("Step 1: Streamlit OK")

import tabpfn_client
st.write("Step 2: tabpfn_client imported")

token = os.environ.get("TABPFN_ACCESS_TOKEN", "")
st.write(f"Step 3: Token found = {bool(token)}, length = {len(token)}")

try:
    tabpfn_client.set_access_token(token)
    st.success("Step 4: set_access_token() succeeded")
except Exception as e:
    st.error(f"Step 4 failed: {e}")
    st.stop()

try:
    from tabpfn_client import TabPFNRegressor
    import numpy as np
    import pandas as pd

    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y = np.array([1.0, 2.0, 3.0])
    m = TabPFNRegressor()
    m.fit(X, y)
    st.success("Step 5: TabPFNRegressor fit succeeded")
except Exception as e:
    st.error(f"Step 5 failed: {e}")
