import streamlit as st
from navigation import render_top_menu, pages, link_2_name
from markdown import outlierFinder, sdtSuite, classification, regionProps
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu on the main page
render_top_menu()

st.title("Index")
titles = [link_2_name(page) for page in pages]
st.markdown("<h3 style='text-align: center;'>Select a step in the Flim Analysis Pipeline to know more</h3>", unsafe_allow_html=True)
col1, col2 = st.columns([0.5, 1])
with col1: 
    selected_step = st.selectbox(
                    "Steps", 
                    titles, 
                    index=0, 
                    key="menu_steps",
    )
with col2: 
    st.markdown("<h4 style='text-align: center;'>Explanation</h4>", unsafe_allow_html=True)
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