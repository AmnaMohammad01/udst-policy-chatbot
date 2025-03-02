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

# Use Streamlit Cache to Store FAISS Index
@st.cache_resource
def load_faiss_index():
    if "faiss_index" in st.session_state:
        return st.session_state["faiss_index"]
    else:
        return None

# Use Session State for Data Persistence
if "all_chunks" not in st.session_state:
    st.session_state["all_chunks"] = []

if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = None

# List of policy URLs
urls = [
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-procedure",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-procedure",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-procedure",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/registration-procedure"
]

# Function to scrape policy content
def fetch_policies(url_list):
    policies = {}
    for url in url_list:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.find("div")  # Modify this selector if needed
            policies[url] = content.get_text(strip=True) if content else "No content found"
        except requests.exceptions.RequestException as e:
            policies[url] = f"Failed to retrieve content. Error: {e}"
    return policies

# Chunking function: Splits text into small parts
def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Get embeddings using Mistral (Batch Processing with Delay)
def get_text_embedding(text_chunks, batch_size=5):
    embeddings_list = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        success = False
        retries = 3  # Retry up to 3 times
        while not success and retries > 0:
            try:
                response = client.embeddings.create(model="mistral-embed", inputs=batch)
                embeddings_list.extend(response.data)
                success = True
            except Exception as e:
                print(f"⚠️ API error: {e} | Retrying in 3 seconds...")
                time.sleep(3)
                retries -= 1
        time.sleep(2)  # Add delay to avoid rate limits
    return embeddings_list

# Load FAISS index if it exists
if load_faiss_index():
    print("🔄 FAISS index exists. Loading...")
    index = load_faiss_index()
    all_chunks = st.session_state["all_chunks"]
else:
    print("🆕 FAISS index does not exist. Generating embeddings...")
    policy_data = fetch_policies(urls)
    all_chunks = []
    for url, text in policy_data.items():
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    text_embeddings = get_text_embedding(all_chunks)
    embeddings = np.array([text_embeddings[i].embedding for i in range(len(text_embeddings))])

    # Create FAISS index in memory (no file storage)
    d = len(text_embeddings[0].embedding)
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # Store in session state
    st.session_state["faiss_index"] = index
    st.session_state["all_chunks"] = all_chunks

    print("✅ FAISS Index successfully created and stored in memory.")

# **Streamlit UI**
st.title("UDST Policy RAG Chatbot")
st.write("Ask questions about UDST policies and get relevant answers.")

question = st.text_input("Enter your question:", "")

if st.button("Get Answer"):
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
