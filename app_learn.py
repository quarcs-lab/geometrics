"""Streamlit Community Cloud entry point for the geometrics **Learn** app.

Deploy this file (or run it locally) to get the Learn module's pages — the concept
sandboxes with sliders and the explainer browser. No dataset needed::

    streamlit run app_learn.py
"""

from geometrics.streamlit_app._entry import run_app

run_app(module="learn")
