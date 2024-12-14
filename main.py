import streamlit as st
from navigation import render_top_menu, pages, link_2_name
from markdown import outlierFinder, sdtSuite, classification, regionProps
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu on the main page
render_top_menu()
deployed_url = "https://flim-playground.streamlit.app/"
github_repo_url = "https://github.com/skalalab/flim_playground"
st.title("Flim Playground")
st.write(f"Welcome! This tool can be run in two modes: **Online** and **Offline**. You can access the online mode by clicking on this [link]({deployed_url}). \
          The **online** mode does not require any setup, but it is slower (it uses some remote, free (crappy) machines kindly offerd by Streamlit). \
         The **offline** mode is faster and more flexible because it runs on your local machine. To run the **offline** mode, you just need to set up a python environment and install the required packages after downloading the \
         necessary files. For detailed instructions, you can come to me or visit the [github repo]({github_repo_url}).")
titles = [link_2_name(page) for page in pages]
st.markdown("<h4 style='text-align: center;'>Select a module to know more</h4>", unsafe_allow_html=True)
col1, col2 = st.columns([0.5, 1])
with col1: 
    selected_step = st.selectbox(
                    "Steps", 
                    titles, 
                    index=0, 
                    key="menu_steps",
    )
with col2: 
    st.markdown("<h5 style='text-align: center;'>Explanation</h5>", unsafe_allow_html=True)
    if selected_step == "Region Props":
        pass
    elif selected_step == "Clustering & Outlier Finder":
        st.markdown(outlierFinder)
    elif selected_step == "Unsupervised Clustering":
        pass
    elif selected_step == "Sdt Suite":
        st.markdown(sdtSuite)
    elif selected_step == "Classification":
        pass
    elif selected_step == "Plotting":
        pass
    
    

# st.write("This is the main page. Use the top menu to navigate to other pages.")