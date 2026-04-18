# Databricks notebook source
# MAGIC %md
# MAGIC # Chat Inference with Databricks Vector Search
# MAGIC
# MAGIC This notebook initializes retrieval from Vector Search and runs question-answer inference using phi-2.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q databricks-vectorsearch sentence-transformers 'langchain>=0.1.0' langchain-core langchain-community transformers torch accelerate

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
dbutils.widgets.text("retrieval_k", "2")
dbutils.widgets.text("max_new_tokens", "512")
dbutils.widgets.text("temperature", "0.8")

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

# COMMAND ----------

# DBTITLE 1,Define prompt template
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# COMMAND ----------

# DBTITLE 1,Load configuration
config = {
    "model_path": dbutils.widgets.get("model_path"),
    "catalog_name": dbutils.widgets.get("catalog_name"),
    "schema_name": dbutils.widgets.get("schema_name"),
    "index_name": dbutils.widgets.get("index_name"),
    "vector_search_endpoint": dbutils.widgets.get("vector_search_endpoint"),
    "retrieval_k": int(dbutils.widgets.get("retrieval_k")),
    "max_new_tokens": int(dbutils.widgets.get("max_new_tokens")),
    "temperature": float(dbutils.widgets.get("temperature")),
}

if not os.path.exists(config["model_path"]):
    raise FileNotFoundError(f"Model directory not found: {config['model_path']}")

for k, v in config.items():
    print(f"{k}: {v}")

# COMMAND ----------

# DBTITLE 1,Initialize retrieval and LLM
# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Vector Search client
vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name=config['vector_search_endpoint'],
    index_name=f"{config['catalog_name']}.{config['schema_name']}.{config['index_name']}"
)

# Create custom retriever for Vector Search
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class VectorSearchRetriever(BaseRetriever):
    index: any
    embeddings: any
    k: int = 2
    
    class Config:
        arbitrary_types_allowed = True
    
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

retriever = VectorSearchRetriever(index=index, embeddings=embeddings, k=config['retrieval_k'])

# Load phi-2 model
print("Loading phi-2 model...")
tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
model = AutoModelForCausalLM.from_pretrained(
    config["model_path"],
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=config['max_new_tokens'],
    temperature=config['temperature'],
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Wrap in LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create prompt
prompt = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
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

print("QA chain initialized with Vector Search and phi-2.")
print("Usage: qa_chain.invoke('Your question here')")

# COMMAND ----------

# DBTITLE 1,Test QA chain
dbutils.widgets.text("question", "What are allergies?")
question = dbutils.widgets.get("question")
result = qa_chain.invoke(question)
print("Question:", question)
print("\nAnswer:", result)