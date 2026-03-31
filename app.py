import os
import streamlit as st

st.set_page_config(page_title="Debug", layout="wide")
st.title("Debug Mode")

# Step 1: 환경변수 확인
token = os.environ.get("TABPFN_ACCESS_TOKEN", "")
st.write(f"Token found: {bool(token)}")
st.write(f"Token length: {len(token)}")

# Step 2: tabpfn_client import 확인
try:
    import tabpfn_client
    st.success(f"tabpfn_client imported. Version: {getattr(tabpfn_client, '__version__', 'unknown')}")
except Exception as e:
    st.error(f"tabpfn_client import failed: {e}")
    st.stop()

# Step 3: init() 확인
try:
    token_dir = os.path.expanduser("~/.tabpfn")
    os.makedirs(token_dir, exist_ok=True)
    with open(os.path.join(token_dir, "token"), "w") as f:
        f.write(token)
    st.success("Token file written.")
except Exception as e:
    st.error(f"Token file write failed: {e}")

try:
    tabpfn_client.init(use_server=True)
    st.success("init() succeeded.")
except TypeError:
    try:
        tabpfn_client.init()
        st.success("init() succeeded (no args).")
    except Exception as e:
        st.error(f"init() failed: {e}")
except Exception as e:
    st.error(f"init() failed: {e}")

# Step 4: 모델 import 확인
try:
    from tabpfn_client import TabPFNRegressor
    st.success("TabPFNRegressor imported.")
except Exception as e:
    st.error(f"TabPFNRegressor import failed: {e}")
