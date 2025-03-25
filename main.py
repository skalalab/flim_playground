import streamlit as st
from navigation import render_top_menu, titles
from docs import docs
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu on the main page
render_top_menu()
left_column, center_column, right_column = st.columns([1.5, 1, 1.5])
# Display the logo in the center column
logo_file = "logo/FP_trans_320.png"
with center_column:
    st.image(logo_file)

deployed_url = "https://flim-playground.streamlit.app/"
github_repo_url = "https://github.com/skalalab/flim_playground"
st.title("Flim Playground")
st.write(f"Welcome! This tool can be run in two modes: **Online** and **Offline**. You can access the online mode by clicking on this [link]({deployed_url}). \
          The **online** mode does not require any setup, but it is **slower** and **NOT secure** (it uses some remote, free (crappy) machines kindly offerd by Streamlit and they can see your data). \
         The **offline** mode is faster and secure because it runs on your local machine. To run the **offline** mode, you just need to set up a python environment and install the required packages after downloading the \
         necessary files. For detailed instructions, you can come to me or visit the [github repo]({github_repo_url}).")

st.markdown("<h4 style='text-align: center;'>Select a playground to know more</h4>", unsafe_allow_html=True)
col1, col2 = st.columns([0.5, 1])
with col1: 
    selected_playground = st.selectbox(
                    "Playgrounds", 
                    titles, 
                    index=0, 
                    key="menu_steps",
    )
with col2: 
    st.markdown("<h5 style='text-align: center;'>Explanation</h5>", unsafe_allow_html=True)

    try:
        doc = docs[selected_playground]
        st.markdown(doc, unsafe_allow_html=True)
    except KeyError:
        st.markdown("<h5 style='text-align: center; color: red'>No doc available yet.</h5>", unsafe_allow_html=True)

    
# st.write("This is the main page. Use the top menu to navigate to other pages.")