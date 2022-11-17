import streamlit as st
from PIL import Image
import classify
import numpy as np

sign_names = {
              0:"With mask",
              1:"Without mask"}

st.title("Face mask Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write("")

        if st.button('predict'):
                st.write("Result...")
                label = classify.predict(uploaded_file)
                label=np.round(label)
                label = label.item()

                res = sign_names.get(label)
                st.markdown(res)
