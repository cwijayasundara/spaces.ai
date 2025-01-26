import streamlit as st
import asyncio
import nest_asyncio
from pathlib import Path
import os
nest_asyncio.apply()
from ingest_documents import ingest_documents
from kb_agents import RagAgent, agent_rag_tool, query_kb

def sanitize_filename(filename):
    """
    Sanitize the filename to prevent path traversal attacks and remove unwanted characters.
    """
    filename = os.path.basename(filename)
    filename = "".join(c for c in filename if c.isalnum() or c in (" ", ".", "_", "-")).rstrip()
    return filename

def save_uploaded_file(uploaded_file, file_path):
    """Save the uploaded file and return the save path"""
    upload_dir = Path.cwd()/"uploaded_docs" / file_path
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_filename = Path(uploaded_file.name).name
    sanitized_filename = sanitize_filename(original_filename)
    save_path = str(upload_dir / sanitized_filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

async def get_answer(query):
    agent = RagAgent(timeout=600, verbose=True)
    handler = agent.run(
        query=query,
        tools=[agent_rag_tool],
    )
    final_result = await handler
    st.write(final_result)

async def get_answer_from_kb(query):
    answer = query_kb(query)
    st.write(answer)

PERSIST_DIR = "./vector_db"

st.title("Welcome to Spaces.ai")
st.write("Agentic system to manage your documents!")

with st.sidebar:
    st.image("images/logo.jpg", width=600)
    add_radio = st.radio(
        "**Select Operation**",
        options=(
            "upload your documents",
            "Chat with your knowledge base",
        )
    )

if add_radio == "upload your documents":
    uploaded_file = st.file_uploader("Choose file", type=["pdf", ".csv", ".docx", ".txt", ".xlsx"])
    file_uploader_button = st.button("Upload", icon="⬆️")
    if uploaded_file and file_uploader_button:
        save_path = save_uploaded_file(uploaded_file, "files")
        ingest_documents(save_path, PERSIST_DIR)
        st.write("your document is successfully saved!")

elif add_radio == "Chat with your knowledge base":
    query = st.text_input("Enter your question")
    if st.button("Answer", icon="💬"):
        if query:
            asyncio.run(get_answer_from_kb(query))
        else:
            st.warning("Please enter a query first.")





