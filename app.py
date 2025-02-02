import streamlit as st
import asyncio
import nest_asyncio
from pathlib import Path
import os
nest_asyncio.apply()
from ingest_documents import ingest_documents
from kb_agents import RagAgent, agent_rag_tool, query_kb
from document_processor import parse_and_summerise_document, list_files_in_directory, generate_sensible_questions

st.set_page_config(layout="wide")  # Enable wide mode for the app

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
SUMMERISED_DIR = "./doc_summery/"
UPLOADED_DIR = "./uploaded_docs/files/"

st.title("Welcome to Spaces.ai")
st.write("Agentic system to manage your documents!")

# Add variables into the Streamlit session state
if "uploaded_file_paths" not in st.session_state:
    st.session_state.uploaded_file_paths = []
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

with st.sidebar:
    st.image("images/logo.jpg", width=300)
    add_radio = st.radio(
        "**Select Operation**",
        options=(
            "home",
            "upload your documents",
            "Chat with your knowledge base",
        )
    )

if add_radio == "home":
    st.write("## Home Page")
    
    # Use full screen layout with columns
    left_col, right_col = st.columns([3, 2])  # Adjust column widths
    
    with left_col:
        st.header("Uploaded Documents")
        files = list_files_in_directory(UPLOADED_DIR)
        if files:
            selected_file = st.radio("Select a file:", options=files, key="selected_file")
            summerise_checkbox = st.checkbox("Summarise")
            questions_checkbox = st.checkbox("Generate Questions")
        else:
            st.write("No uploaded documents.")
        
    with right_col:
        if selected_file and summerise_checkbox:
            st.header("Document Summarised")
            summerised_text = parse_and_summerise_document(selected_file)
            st.write(summerised_text)
        elif selected_file and questions_checkbox:
            st.header("Document Questions")
            questions_df = generate_sensible_questions(selected_file)
            st.dataframe(questions_df)  # Use a table for better visualization

elif add_radio == "upload your documents":
    st.write("## Upload Your Documents")
    uploaded_file = st.file_uploader("Choose file", type=["pdf", ".csv", ".docx", ".txt", ".xlsx"])
    file_uploader_button = st.button("Upload")
    if uploaded_file and file_uploader_button:
        save_path = save_uploaded_file(uploaded_file, "files")
        ingest_documents(save_path, PERSIST_DIR)
        st.session_state.uploaded_file_paths.append(save_path)
        st.success("Your document has been successfully saved!")

elif add_radio == "Chat with your knowledge base":
    st.write("## Chat with Your Knowledge Base")
    query = st.text_input("Enter your question")
    if st.button("Answer"):
        if query:
            asyncio.run(get_answer_from_kb(query))
        else:
            st.warning("Please enter a query first.")
