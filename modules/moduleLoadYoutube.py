# --- / 
import time
from typing import Callable, Optional

import cv2
import streamlit as st
from cap_from_youtube import cap_from_youtube


def youtube(loadedModel:object,functionToRun:Callable, url:str,objectClasses:list,requiredConfidence:float) -> Optional[bool]:
    vid_cap = cap_from_youtube(url)

    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st1, st2, st3 = st.columns(3)
    with st1:
        st.markdown("## Height")
        st1_text = st.markdown(f"{height}")
    with st2:
        st.markdown("## Width")
        st2_text = st.markdown(f"{width}")
    with st3:
        st.markdown("## FPS")
        st3_text = st.markdown("0")

    # Setting divider
    st.markdown("---")

    # Output image
    st_frame = st.empty()
    
    # loading model before use ! 
    terminatedVideoStream = functionToRun(loadedModel,vid_cap,st_frame,st3_text,objectClasses,requiredConfidence)

    if terminatedVideoStream:
        # Close the stream
        vid_cap.release()
