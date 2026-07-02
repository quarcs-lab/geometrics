"""Streamlit Community Cloud entry point for the geometrics **Explore** app.

Deploy this file (or run it locally) to get the Explore module's pages — maps,
weights connectivity, Moran/LISA, and space-time views — on the bundled case studies::

    streamlit run app_explore.py
"""

from geometrics.streamlit_app._entry import run_app

run_app(module="explore")
