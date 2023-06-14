# --- / 
# -- / external imports 
import glob

import time
import cv2
# import pafy
import PIL
import streamlit as st
from PIL import Image






# Load the YOLO model








# Does not work yet
""" def youtube():
    source_youtube = st.sidebar.text_input("YouTube url")
    video = pafy.new(source_youtube)
    best = video.getbest(preftype="mp4")
    vid_cap = cv2.VideoCapture(best.url)
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st_frame = st.empty()
    fps = 0
    st1, st2, st3 = st.columns(3)
    with st1:
        st.markdown("## Height")
        st1_text = st.markdown(f"{height}")
    with st2:
        st.markdown("## Width")
        st2_text = st.markdown(f"{width}")
    with st3:
        st.markdown("## FPS")
        st3_text = st.markdown(f"{fps}")
    while(vid_cap.isOpened()):
        ret, frame = vid_cap.read()
        if not ret:
            st.write("Can't read frame, stream ended? Exiting ....")
            break
        frame = cv2.resize(frame, (720, int(720*(9/16))))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = model.predict(frame, conf=confidence, classes=classes)
        res_plotted = res[0].plot()
        st_frame.image(res_plotted, caption="Detected Video", use_column_width=True)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        st1_text.markdown(f"**{height}**")
        st2_text.markdown(f"**{width}**")
        st3_text.markdown(f"**{fps:.2f}**")
    vid_cap.release() """


# Offline Data loading





