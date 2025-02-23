from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_parse import LlamaParse
import os
from typing import List
import pandas as pd

_ = load_dotenv()

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
resoning_llm = OpenAI(model="o3-mini", temperature=1.0, reasoning_effort="medium")

parser = LlamaParse(
    result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="gemini-2.0-flash-001",
    invalidate_cache=True,
    parsing_instruction="",
)

summerised_document_path = "./doc_summery"
questions_path = "./generated_questions/"

def parse_document(file_path: str) -> str:
    """Parse the document and return the text content."""
    documents = parser.load_data(file_path)
    # convert the documents to a string
    text = "\n".join([doc.text for doc in documents])
    return text

def generate_sensible_questions(file_path: str) -> List[str]:
    """Generate a list of sensible questions from the text."""
    text = parse_document(file_path)
    prompt = PromptTemplate(
        template="Generate 10 sensible questions from the following text: {text}. Do not return any other text than the questions.",
        complemental_formatting_instruction=""
    )
    questions = str(llm.complete(prompt.format(text=text)))
    # add questions to a pandas dataframe
    questions_df = pd.DataFrame({"questions": questions.split("\n")})
    # remove any null questions or empty questions or questions that are not strings
    questions_df = questions_df[questions_df["questions"].notna()]
    # save the questions to a csv file
    os.makedirs(questions_path, exist_ok=True)
    questions_df.to_csv(os.path.join(questions_path, os.path.basename(file_path) + ".csv"), index=False)
    return questions_df

def summerise_document(text: str) -> str:
    """Summerise the document using the LLM."""
    prompt = PromptTemplate(
        template="Summarise the following document without losing any important information: {text}",
        complemental_formatting_instruction=""
    )
    return str(llm.complete(prompt.format(text=text)))

def parse_and_summerise_document(file_path: str) -> str:
    """Parse the document and summerise it."""
    text = parse_document(file_path)
    summerised_text = summerise_document(text)
    # Ensure the summerised directory exists
    os.makedirs(summerised_document_path, exist_ok=True)
    # Save the summerised text to a file using os.path.join as a .md file
    with open(os.path.join(summerised_document_path, os.path.basename(file_path) + ".md"), "w") as file:
        file.write(summerised_text)
    return summerised_text

# function to list all the files with relative path
def list_files_in_directory(directory_path: str) -> List[str]:
    return [os.path.join(directory_path, file) for file in os.listdir(directory_path)]

