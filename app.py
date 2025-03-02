import streamlit as st
import os
import time
import requests
import numpy as np
import faiss
import pickle
from bs4 import BeautifulSoup
from mistralai import Mistral
from mistralai.models import UserMessage

# Load API Key from Streamlit Secrets
API_KEY = st.secrets["MISTRAL_API_KEY"]
client = Mistral(api_key=API_KEY)

# Define policy URLs and names
policies = {
    "Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
    "Graduate Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
    "Graduate Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-policy",
    "Graduate Academic Standing Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-procedure",
    "Graduate Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
    "Graduate Final Grade Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-procedure",
    "International Student Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
    "International Student Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-procedure",
    "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
    "Registration Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/registration-procedure"
}

# **Streamlit UI Styling**
st.set_page_config(page_title="UDST Policy RAG Chatbot", layout="wide")
st.markdown("""
    <style>
        body { background-color: #1e1e1e; color: white; }
        .main { background-color: #262626; padding: 20px; border-radius: 10px; }
        .stTextInput, .stTextArea, .stButton { border-radius: 10px; }
        .stButton button { background-color: #ff4b4b; color: white; border-radius: 10px; font-size: 14px; }
        .policy-container { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
        .policy-container a { background-color: #f0f0f0; padding: 6px 10px; border-radius: 5px; text-decoration: none; color: #333; font-size: 14px; text-align: center; display: block; }
    </style>
""", unsafe_allow_html=True)

# **Title Section**
st.title("UDST Policy RAG Chatbot")
st.write("Ask questions about UDST policies and get relevant answers.")

# **Policy List as Compact Hyperlinks in Two Rows**
st.subheader("Available Policies")
st.markdown('<div class="policy-container">' + ''.join([f'<a href="{url}" target="_blank">{policy}</a>' for policy, url in policies.items()]) + '</div>', unsafe_allow_html=True)

# **User Query Section**
st.subheader("Ask a Question")
question = st.text_input("Enter your question:", "", key="question_input")

if st.button("Get Answer", key="get_answer_button"):
    if question:
        question_embedding = np.array([get_text_embedding([question])[0].embedding])
        D, I = index.search(question_embedding, k=2)
        retrieved_chunks = [all_chunks[i] for i in I.tolist()[0]]

        prompt = f"""
        Context information is below.
        ---------------------
        {retrieved_chunks}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {question}
        Answer:
        """
        messages = [UserMessage(content=prompt)]
        response = client.chat.complete(model="mistral-large-latest", messages=messages)
        answer = response.choices[0].message.content if response.choices else "No response generated."

        st.text_area("Answer:", answer, height=200)
    else:
        st.warning("Please enter a question.")
