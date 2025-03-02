import streamlit as st

# Set page config at the very beginning
st.set_page_config(page_title="UDST Policy Chatbot", layout="wide")

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

# Function to fetch policy data
def fetch_policies():
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
    return policies

# Function to regenerate FAISS index
def regenerate_embeddings():
    st.info("Regenerating FAISS index and embeddings. Please wait...")
    policies = fetch_policies()
    all_chunks = []
    for url in policies.values():
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.get_text(strip=True)
            all_chunks.append(content)
        except Exception as e:
            st.error(f"Error fetching {url}: {e}")

    embeddings = [client.embeddings.create(model="mistral-embed", inputs=[chunk]).data[0].embedding for chunk in all_chunks]
    embeddings = np.array(embeddings)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(ALL_CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    
    st.success("FAISS index and embeddings successfully regenerated.")
    return index, all_chunks

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
    st.warning("FAISS index and policy data not found.")
    if st.button("Regenerate Embeddings"):
        index, all_chunks = regenerate_embeddings()

# Function to get text embeddings
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

# **Title Section**
st.title("UDST Policy Chatbot")
st.write("Ask questions about UDST policies and get relevant answers.")

# **User Query Section**
st.subheader("Ask a Question")
question = st.text_input("Enter your question:", "", key="question_input")

if st.button("Get Answer", key="get_answer_button"):
    if question:
        if index is None or all_chunks is None:
            st.error("FAISS index is not available. Please regenerate embeddings.")
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
