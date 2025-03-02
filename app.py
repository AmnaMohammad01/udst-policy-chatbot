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
try:
    API_KEY = st.secrets["MISTRAL_API_KEY"]
    if not API_KEY:
        raise ValueError("API Key not found in Streamlit secrets.")
except Exception as e:
    st.error(f"Error loading API Key: {e}")
    st.stop()

client = Mistral(api_key=API_KEY)

# Define paths for FAISS index and chunks
FAISS_INDEX_PATH = "policy_embeddings.index"
ALL_CHUNKS_PATH = "all_chunks.pkl"

# Load FAISS index and all_chunks if available
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(ALL_CHUNKS_PATH):
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(ALL_CHUNKS_PATH, "rb") as f:
            all_chunks = pickle.load(f)
        st.success("Policy data loaded successfully.")
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        index, all_chunks = None, None
else:
    st.warning("FAISS index and policy data not found. Please regenerate embeddings.")
    index, all_chunks = None, None

# **Function to get text embeddings**
def get_text_embedding(text_chunks):
    embeddings_list = []
    for text in text_chunks:
        try:
            response = client.embeddings.create(model="mistral-embed", inputs=[text])
            embeddings_list.append(response.data[0].embedding)
        except Exception as e:
            st.error(f"Error fetching embeddings: {e}")
            return None
    return embeddings_list

# **Streamlit UI Styling**
st.set_page_config(page_title="UDST Policy Chatbot", layout="wide")
st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #874F41; color: white;
        }
        .main { background-color: #6e3d32; padding: 20px; border-radius: 10px; }
        .stTextInput, .stTextArea, .stButton { border-radius: 10px; }
        .stButton button { background-color: #ff4b4b; color: white; border-radius: 10px; font-size: 14px; }
        .policy-container { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; justify-content: center; }
        .policy-container a { background-color: #f0f0f0; padding: 6px 10px; border-radius: 5px; text-decoration: none; color: #333; font-size: 14px; text-align: center; display: block; }
    </style>
""", unsafe_allow_html=True)

# **Title Section**
st.title("UDST Policy Chatbot")
st.write("Ask questions about UDST policies and get relevant answers.")

# **User Query Section**
st.subheader("Ask a Question")
question = st.text_input("Enter your question:", "", key="question_input")

if st.button("Get Answer", key="get_answer_button"):
    if question:
        if index is None or all_chunks is None:
            st.error("FAISS index is not available. Please check your data.")
        else:
            question_embedding = get_text_embedding([question])
            if question_embedding is None:
                st.error("Failed to generate embeddings. Please try again later.")
            else:
                question_embedding = np.array(question_embedding)
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
                try:
                    response = client.chat.complete(model="mistral-large-latest", messages=messages)
                    answer = response.choices[0].message.content if response.choices else "No response generated."
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    answer = "Error: Could not generate response."
                st.text_area("Answer:", answer, height=200)
    else:
        st.warning("Please enter a question.")
