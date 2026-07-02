"""Streamlit Community Cloud entry point for the geometrics **Analyze** app.

Deploy this file (or run it locally) to get the Analyze module's pages — convergence,
spatial models with impacts, dynamics, and inequality — on the bundled case studies::

    streamlit run app_analyze.py
"""

from geometrics.streamlit_app._entry import run_app

run_app(module="analyze")
