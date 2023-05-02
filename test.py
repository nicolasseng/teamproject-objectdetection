import pandas as pd
import streamlit as st


def main():
    st.sidebar.title("...")
    app_mode = st.sidebar.selectbox("Choose smth", ["Run", "Code"])


def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.01
    )
    overlap_threshold = st.sidebar.slider("Overlap_threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold


if __name__ == "__main__":
    main()
