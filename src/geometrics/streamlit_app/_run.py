"""The packaged script the launchers point ``streamlit run`` at.

The module to show is handed over via the ``GEOMETRICS_MODULE`` environment variable.
"""

from geometrics.streamlit_app._entry import run_app

run_app()
