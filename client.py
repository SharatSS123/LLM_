import requests
import os
import streamlit as st 

st.title('Personal query bot')

url= "http://localhost:8800/"

uploaded_file = st.file_uploader("Upload PDF File", type="pdf")

if st.button("upload"):
    if uploaded_file:


    # st.write("File Details:")
    # st.write(f"Name: {uploaded_file.name}")
    # st.write(f"Size: {uploaded_file.size} bytes")

        file_path = os.path.join('/Users/lynn/Documents/Sharat/Dev/app', uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        # if st.button("upload"):
        output=requests.post(url+'upload', params={"file_name":  f"{uploaded_file.name}"})
        st.write(output.json())

input_text=st.text_input("Ask your query")

if st.button("search"):
    if input_text:
        print(input_text)
        input_req=input_text.encode("utf-8")
        output=requests.post(url+'search', params={"input_text": input_req })
        st.write(output.json())