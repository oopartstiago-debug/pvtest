import streamlit as st
st.set_page_config(page_title="Debug")
st.title("Step 1: Streamlit OK")

import tabpfn_client
st.write("Step 2: tabpfn_client imported")
st.write(dir(tabpfn_client))
