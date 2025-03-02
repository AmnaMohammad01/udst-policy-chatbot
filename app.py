import time
import requests
import numpy as np
import faiss
import streamlit as st
from bs4 import BeautifulSoup
from mistralai import Mistral
from mistralai.models import UserMessage
import pickle

# Load API Key from Streamlit Secrets
API_KEY = st.secrets["MISTRAL_API_KEY"]
client = Mistral(api_key=API_KEY)

# FAISS & Data Paths
FAISS_INDEX_PATH = "policy_embeddings.index"
ALL_CHUNKS_PATH = "all_chunks.pkl"

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
    client = Mistral(api_key=API_KEY)
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

# **Load FAISS & all_chunks if available, otherwise create embeddings**
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(ALL_CHUNKS_PATH):
    print("🔄 FAISS index and all_chunks exist. Loading...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(ALL_CHUNKS_PATH, "rb") as f:
        all_chunks = pickle.load(f)
    print("✅ Successfully loaded FAISS index and all_chunks.")
else:
    print("🆕 FAISS index does not exist. Generating embeddings...")
    policy_data = fetch_policies(urls)
    all_chunks = []
    for url, text in policy_data.items():
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    text_embeddings = get_text_embedding(all_chunks)
    embeddings = np.array([text_embeddings[i].embedding for i in range(len(text_embeddings))])
    d = len(text_embeddings[0].embedding)  # Embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(ALL_CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    print("✅ FAISS Index and all_chunks successfully created and saved.")

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
