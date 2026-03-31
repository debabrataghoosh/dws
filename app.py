import streamlit as st

st.set_page_config(page_title="Drowsiness Detection System", page_icon="😴", layout="centered")

st.title("Drowsiness Detection System 😴")
st.write("Click button to start detection")

if st.button("Start Camera"):
    st.write("Camera will start here...")
    st.info(
        "This Streamlit demo is UI-only. The full webcam + OpenCV window pipeline "
        "runs locally from drowsiness_detector.py."
    )

st.markdown("---")
st.subheader("Reality Check")
st.warning(
    "Streamlit Community Cloud cannot directly run this project's cv2 webcam window "
    "pipeline, and dlib deployments are commonly problematic."
)

st.table(
    {
        "Feature": ["Webcam (cv2)", "dlib", "Basic UI"],
        "Works on Streamlit Cloud?": ["No", "No", "Yes"],
    }
)

st.caption("Run full detection locally with: python drowsiness_detector.py")
