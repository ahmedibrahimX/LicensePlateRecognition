import os
from skimage import io
from preprocessing import Preprocessing
from detect import LicenceDetection
import streamlit as st

directory_path = "./images"
images = os.listdir(directory_path)
image_name = st.sidebar.selectbox("Select a car image:", ["sample1.jpg", "sample2.jpg"], index=0)

@st.cache
def pipeline_execution():
    preprossed_img, gray_img = Preprocessing.preprocess_photo(img)
    lp, lpr, segmented_char, ocr_output = LicenceDetection.license_detect(preprossed_img, gray_img)
    return lp, lpr, segmented_char, ocr_output

if image_name:
    img = io.imread(directory_path+"/"+image_name)
    st.subheader("License Plate Recognition")
    st.subheader("Selected Image:")
    st.image(img, channels="RGB")
    st.markdown("---")
    
    st.sidebar.text("LPR Progress:")
    my_bar = st.sidebar.progress(0)
    with st.spinner('Preparing plate...'):
        lp, lpr, segmented_char, ocr_output = pipeline_execution()
    
    if lp is not None:
        if st.sidebar.checkbox("Show plate"):
            my_bar.progress(25)
            st.subheader("Detected Plate:")
            st.image(lp, use_column_width=True, clamp = True)
            st.markdown("---")
            if st.sidebar.checkbox("Binarize"):
                my_bar.progress(50)
                st.subheader("Binarized Plate:")
                st.image(lpr, use_column_width=True, clamp = True)
                st.markdown("---")
                if st.sidebar.checkbox("Show segmented characters"):
                    my_bar.progress(75)
                    st.subheader("Segmented Characters:")
                    st.image(segmented_char, use_column_width=True, clamp = True)
                    st.markdown("---")
                    if st.sidebar.checkbox("Show OCR output"):
                        my_bar.progress(100)
                        st.subheader("OCR output:")
                        st.markdown("**" + ocr_output + "**")
                        st.balloons()
                        st.markdown("---")