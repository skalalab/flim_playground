# FLIM Playground


# Quick try 
It is deployed at: [https://flim-playground.streamlit.app/](https://flim-playground.streamlit.app/). 
You can try out the Visualization and Classification modules using this sample [dataset] extracted by Data Extraction module

# Install
## Download from Releases
- Releases for Mac OS 15, Windows 11
## Build from source
## Clone the repo
Navigate inside the repo once cloned. 
## Install the python environment
- Install `uv` if not yet installed
- run `uv sync`
## Build
streamlit-desktop-app build main.py --name Flim-Playground  --icon logo.png --pyinstaller-options --add-data src:src --add-data pages:pages --add-data config.toml:. --add-data logo.png:. --hidden-import pages.classification --hidden-import pages.visualization --hidden-import pages.data_extraction --collect-all streamlit_plotly_events --onefile --noconfirm --clean

# Documentation
- @[docs]()


# TODO
- more rigorous classification 