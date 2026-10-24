"""
Streamlit Cloud Entry Point
Delegates execution to app.py so both 'app.py' and 'streamlit_app.py' entry points work.
"""
import os
import runpy

app_path = os.path.join(os.path.dirname(__file__), "app.py")
runpy.run_path(app_path, run_name="__main__")
