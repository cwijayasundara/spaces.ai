import nest_asyncio
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import (
    step,
    Event,
    Context,
    StartEvent,
    StopEvent,
    Workflow,
)
from llama_index.core.agent import FunctionCallingAgent
from pathlib import Path

nest_asyncio.apply()

_ = load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
                      
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
    
PERSIST_DIR = "./vector_db"

def get_query_engine():
    """Initialize and return the query engine"""
    if not Path(PERSIST_DIR).exists():
        raise ValueError("No index found. Please ingest documents first.")
        
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    return index.as_query_engine(similarity_top_k=3)

async def search_documents(query: str) -> str:
    """Search through the documents for relevant information."""
    query_engine = get_query_engine()
    response = query_engine.query(query)
    return str(response)

# Create the tool with proper metadata
agent_rag_tool = FunctionTool.from_defaults(
    fn=search_documents,
    name="document_search",
    description="Searches and retrieves information from the uploaded documents",
)

def query_kb(query: str) -> str:
    """Query the knowledge base for the answer to the question."""
    query_engine = get_query_engine()
    response = query_engine.query(query)
    return str(response)

class RagAgent(Workflow):

    @step()
    async def answer_from_kb(
        self, ctx: Context, ev: StartEvent
    ) -> StopEvent:
        query = ev.query
        await ctx.set("original_query", query)
        await ctx.set("tools", ev.tools)

        prompt = f"""You are an expert at answering questions based on the knowledge base.
            You have been given a question to answer. Answer the question: {query}
            Answer the query based on the documents in the context. If you are not sure, say "information not found in the knowledge base".
            """

        response = await Settings.llm.acomplete(prompt)

        return StopEvent(result=str(response))