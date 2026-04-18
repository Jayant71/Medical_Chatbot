# Databricks notebook source
# MAGIC %md
# MAGIC # Smoke Tests
# MAGIC
# MAGIC This notebook validates model file access, Vector Search retrieval, and response generation with phi-2.

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

# DBTITLE 1,Load config and verify model
config = {
    "model_path": dbutils.widgets.get("model_path"),
    "catalog_name": dbutils.widgets.get("catalog_name"),
    "schema_name": dbutils.widgets.get("schema_name"),
    "index_name": dbutils.widgets.get("index_name"),
    "vector_search_endpoint": dbutils.widgets.get("vector_search_endpoint"),
}

assert os.path.exists(config["model_path"]), f"Model path missing: {config['model_path']}"
print("✓ PASS: model directory exists")

# COMMAND ----------

# DBTITLE 1,Test Vector Search retrieval
# Initialize embeddings and Vector Search client
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vsc = VectorSearchClient()

# Get the index
index = vsc.get_index(
    endpoint_name=config['vector_search_endpoint'],
    index_name=f"{config['catalog_name']}.{config['schema_name']}.{config['index_name']}"
)

# Test retrieval
query_embedding = embeddings.embed_query("What are allergies?")
results = index.similarity_search(
    query_vector=query_embedding,
    columns=["id", "text"],
    num_results=2
)

assert len(results['result']['data_array']) > 0, "No retrieval documents returned"
print(f"✓ PASS: retrieval returned {len(results['result']['data_array'])} docs")

# COMMAND ----------

# DBTITLE 1,Test QA chain with phi-2
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
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)

# Prompt template
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
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

answer = qa_chain.invoke("What is acne?")
assert isinstance(answer, str) and len(answer.strip()) > 0, "Empty response"
print("✓ PASS: inference produced non-empty answer")
print(answer[:400])