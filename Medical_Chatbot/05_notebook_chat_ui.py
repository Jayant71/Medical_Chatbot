# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Chat UI (ipywidgets)
# MAGIC
# MAGIC This notebook provides an interactive chat user interface directly inside Databricks notebooks.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q ipywidgets==8.1.2 databricks-vectorsearch sentence-transformers 'langchain>=0.1.0' langchain-core langchain-community transformers torch accelerate pypdf

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configure widgets
dbutils.widgets.text(
    "model_path",
    "/Volumes/medical_chatbot/medical_data/medical_data/model/",
)
dbutils.widgets.text("catalog_name", "medical_chatbot")
dbutils.widgets.text("schema_name", "medical_data")
dbutils.widgets.text("index_name", "medical_bot_index")
dbutils.widgets.text("vector_search_endpoint", "medical_chatbot_endpoint")

# COMMAND ----------

# DBTITLE 1,Import libraries
import os
import torch
import ipywidgets as widgets
from IPython.display import display
from databricks.vector_search.client import VectorSearchClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.schema import BaseRetriever, Document
from typing import List

# COMMAND ----------

# DBTITLE 1,Initialize chat backend
config = {
    "model_path": dbutils.widgets.get("model_path"),
    "catalog_name": dbutils.widgets.get("catalog_name"),
    "schema_name": dbutils.widgets.get("schema_name"),
    "index_name": dbutils.widgets.get("index_name"),
    "vector_search_endpoint": dbutils.widgets.get("vector_search_endpoint"),
}

if not os.path.exists(config["model_path"]):
    raise FileNotFoundError(f"Model path missing: {config['model_path']}")

# Prompt template
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Initialize embeddings and Vector Search
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name=config['vector_search_endpoint'],
    index_name=f"{config['catalog_name']}.{config['schema_name']}.{config['index_name']}"
)

# Custom retriever for Vector Search
class VectorSearchRetriever(BaseRetriever):
    def __init__(self, index, embeddings, k=2):
        self.index = index
        self.embeddings = embeddings
        self.k = k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.similarity_search(
            query_vector=query_embedding,
            columns=["id", "text"],
            num_results=self.k
        )
        docs = []
        for row in results['result']['data_array']:
            docs.append(Document(page_content=row[1], metadata={"id": row[0]}))
        return docs
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

retriever = VectorSearchRetriever(index, embeddings, k=2)

# Load phi-2 model
print("Loading phi-2 model...")
tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
model = AutoModelForCausalLM.from_pretrained(
    config["model_path"],
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.8,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)

# Create prompt
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Helper function to format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create QA chain using LCEL
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✓ Chat backend initialized with Vector Search and phi-2.")

# COMMAND ----------

# DBTITLE 1,Display chat UI
chat_history = widgets.Output(layout={"border": "1px solid #ccc", "padding": "10px"})
question_box = widgets.Text(
    placeholder="Ask a medical question...",
    description="You:",
    layout=widgets.Layout(width="85%"),
)
send_button = widgets.Button(description="Send", button_style="primary")
clear_button = widgets.Button(description="Clear", button_style="warning")
status = widgets.HTML(value="<b>Status:</b> Ready")


def handle_send(_):
    question = question_box.value.strip()
    if not question:
        return

    status.value = "<b>Status:</b> Generating answer..."
    with chat_history:
        print(f"You: {question}")
    try:
        answer = qa_chain.invoke(question)
        with chat_history:
            print(f"Bot: {answer}\n")
        status.value = "<b>Status:</b> Ready"
    except Exception as e:
        with chat_history:
            print(f"Bot Error: {str(e)}\n")
        status.value = "<b>Status:</b> Error"
    question_box.value = ""


def handle_clear(_):
    chat_history.clear_output()
    status.value = "<b>Status:</b> Cleared"


send_button.on_click(handle_send)
clear_button.on_click(handle_clear)

ui = widgets.VBox(
    [
        widgets.HTML(value="<h3>Medical Chatbot UI</h3>"),
        status,
        chat_history,
        widgets.HBox([question_box, send_button, clear_button]),
    ]
)

display(ui)