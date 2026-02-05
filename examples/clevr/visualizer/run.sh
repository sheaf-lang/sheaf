#!/bin/bash

if ! python -c "import streamlit" 2>/dev/null; then
    echo "Streamlit not found."
    echo "Install dependancies with: pip install streamlit matplotlib data"
    exit 1
fi

echo "Starting dashboard at http://localhost:8501"
streamlit run visualizer.py
