
"""
A dictionary of documentations, one for each playground. When a new playground is added,
the corresponding documentation should be added here as well.
"""
deployed_url = "https://flim-playground.streamlit.app/"
github_repo_url = "https://github.com/skalalab/flim_playground"

dataExtraction = """to be developed. """
visualization = """to be developed. """
classification = """to be developed."""
generalInfo = """This tool can be run in two modes: **Online** and **Offline**. You can access the online mode by clicking on this [link]({deployed_url}). The source code is hosted at [Github]({github_repo_url}). \
The **online** mode does not require any setup, but it is **slower** and **NOT secure** (it uses some remote, free (crappy) machines kindly offerd by Streamlit and they can see your data). \
The **offline** mode is faster and secure because it runs on your local machine. To run the **offline** mode, you just need to set up a python environment and install the required packages after downloading the \
necessary files."""

docs = {"Data Extraction": dataExtraction, "Visualization": visualization, "Classification": classification, "General Info": generalInfo}