# Databricks notebook source
# MAGIC %md
# MAGIC # Build Databricks Vector Search Index
# MAGIC
# MAGIC This notebook ingests the medical PDF, creates embeddings, and builds a Databricks Vector Search index.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q databricks-vectorsearch sentence-transformers langchain langchain-community langchain-text-splitters pypdf

# COMMAND ----------

# DBTITLE 1,Configure widgets
dbutils.widgets.text(
    "pdf_path", "/Volumes/medical_chatbot/medical_data/medical_data/Medical_book.pdf"
)
dbutils.widgets.text("catalog_name", "medical_chatbot")
dbutils.widgets.text("schema_name", "medical_data")
dbutils.widgets.text("index_name", "medical_bot_index")
dbutils.widgets.text("vector_search_endpoint", "medical_chatbot_endpoint")
dbutils.widgets.text("chunk_size", "500")
dbutils.widgets.text("chunk_overlap", "20")

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import libraries
import os
from databricks.vector_search.client import VectorSearchClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# COMMAND ----------

# DBTITLE 1,Load configuration
config = {
    "pdf_path": dbutils.widgets.get("pdf_path"),
    "catalog_name": dbutils.widgets.get("catalog_name"),
    "schema_name": dbutils.widgets.get("schema_name"),
    "index_name": dbutils.widgets.get("index_name"),
    "vector_search_endpoint": dbutils.widgets.get("vector_search_endpoint"),
    "chunk_size": int(dbutils.widgets.get("chunk_size")),
    "chunk_overlap": int(dbutils.widgets.get("chunk_overlap")),
}

for k, v in config.items():
    print(f"{k}: {v}")

# COMMAND ----------

def load_pdf_file(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# COMMAND ----------

# DBTITLE 1,Load and process PDF (test with first 10 pages)
if not os.path.exists(config["pdf_path"]):
    raise FileNotFoundError(f"PDF file not found: {config['pdf_path']}")

# Load full PDF
documents = load_pdf_file(config["pdf_path"])

chunks = split_documents(
    documents,
    chunk_size=config["chunk_size"],
    chunk_overlap=config["chunk_overlap"],
)
embeddings = create_embeddings()

print(f"📄 Processing full PDF")
print(f"Loaded pages: {len(documents)}")
print(f"Created chunks: {len(chunks)}")

# COMMAND ----------

# DBTITLE 1,Clean up existing index and table
# Drop existing index and table to start fresh
table_name = f"{config['catalog_name']}.{config['schema_name']}.{config['index_name']}_docs"
index_name = f"{config['catalog_name']}.{config['schema_name']}.{config['index_name']}"

vsc = VectorSearchClient()

# Try to delete existing index
try:
    vsc.delete_index(endpoint_name=config['vector_search_endpoint'], index_name=index_name)
    print(f"✓ Deleted existing index: {index_name}")
except Exception as e:
    print(f"No existing index to delete (this is fine): {e}")

# Try to drop existing table
try:
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    print(f"✓ Dropped existing table: {table_name}")
except Exception as e:
    print(f"No existing table to drop (this is fine): {e}")

print("\n✓ Ready to create fresh index with test data!")

# COMMAND ----------

# DBTITLE 1,Create Vector Search index
# Generate embeddings for all chunks
print("Generating embeddings...")
texts = [chunk.page_content for chunk in chunks]
embeddings_list = embeddings.embed_documents(texts)

# Create DataFrame with id, text, and text_vector columns
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

schema = StructType([
    StructField("id", StringType(), False),
    StructField("text", StringType(), False),
    StructField("text_vector", ArrayType(FloatType()), False)
])

data = [
    (str(i), text, list(map(float, embedding)))
    for i, (text, embedding) in enumerate(zip(texts, embeddings_list))
]

df = spark.createDataFrame(data, schema)

# Create Delta table with Change Data Feed enabled
table_name = f"{config['catalog_name']}.{config['schema_name']}.{config['index_name']}_docs"
df.write.format("delta").mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(table_name)
print(f"Created Delta table: {table_name}")

# Create Vector Search index
vsc = VectorSearchClient()

# Create or update the delta sync index
index = vsc.create_delta_sync_index(
    endpoint_name=config['vector_search_endpoint'],
    index_name=f"{config['catalog_name']}.{config['schema_name']}.{config['index_name']}",
    source_table_name=table_name,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_dimension=384,  # all-MiniLM-L6-v2 has 384 dimensions
    embedding_vector_column="text_vector"
)

print(f"\n✓ Vector Search index created successfully!")
print(f"Index name: {config['index_name']}")
print(f"\nThe index is now initializing. It will take a few minutes to become ready.")
print(f"You can check the status in the Databricks Vector Search UI.")

# COMMAND ----------

# DBTITLE 1,Test similarity search
import time

# Wait for the index to be ready
check_query = "What are allergies?"
max_wait_time = 300  # 5 minutes
start_time = time.time()

print("Waiting for index to be ready...")

index = vsc.get_index(
    endpoint_name=config['vector_search_endpoint'],
    index_name=f"{config['catalog_name']}.{config['schema_name']}.{config['index_name']}"
)

while time.time() - start_time < max_wait_time:
    try:
        status = index.describe()
        index_status = status.get('status', {}).get('indexed_row_count', 0)
        
        print(f"Index status: {status.get('status', {}).get('message', 'Initializing...')}")
        print(f"Indexed rows: {index_status}")
        
        if index_status > 0:
            print("\n✓ Index is ready!\n")
            
            # Test similarity search
            query_embedding = embeddings.embed_query(check_query)
            results = index.similarity_search(
                query_vector=query_embedding,
                columns=["id", "text"],
                num_results=2
            )
            
            print(f"Retrieved {len(results['result']['data_array'])} docs for query: '{check_query}'")
            for i, doc in enumerate(results['result']['data_array'], start=1):
                print(f"\n--- Doc {i} ---")
                print(f"Text preview: {doc[1][:300]}...")
            break
    except Exception as e:
        print(f"Waiting... ({int(time.time() - start_time)}s elapsed)")
    
    time.sleep(10)
else:
    print(f"\nIndex is still initializing after {max_wait_time}s. Check the Vector Search UI for status.")

# COMMAND ----------

