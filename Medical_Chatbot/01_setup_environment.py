# Databricks notebook source
# MAGIC %md
# MAGIC # Medical Chatbot - Databricks Setup
# MAGIC
# MAGIC This notebook prepares the environment for the medical chatbot workflow:
# MAGIC - Installs required dependencies (transformers, langchain, databricks-vectorsearch)
# MAGIC - Downloads and saves the microsoft/phi-2 model to Unity Catalog Volumes
# MAGIC - Configures Databricks Vector Search settings
# MAGIC - Verifies all components are ready

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q databricks-vectorsearch sentence-transformers langchain pypdf python-dotenv transformers torch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-2"
save_path = "/Volumes/medical_chatbot/medical_data/medical_data/model/"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

# COMMAND ----------

!ls /Volumes/medical_chatbot/medical_data/medical_data/model/

# COMMAND ----------

# DBTITLE 1,Verify model can be loaded
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/Volumes/medical_chatbot/medical_data/medical_data/model/"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print("✓ Model and tokenizer loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
except Exception as e:
    print(f"✗ Error loading model: {str(e)}")

# COMMAND ----------

# DBTITLE 1,Configure paths and settings
dbutils.widgets.text(
    "pdf_path", "/Volumes/medical_chatbot/medical_data/medical_data/Medical_book.pdf"
)
dbutils.widgets.text(
    "model_path",
    "/Volumes/medical_chatbot/medical_data/medical_data/model/",
)
dbutils.widgets.text("catalog_name", "medical_chatbot")
dbutils.widgets.text("schema_name", "medical_data")
dbutils.widgets.text("index_name", "medical_bot_index")
dbutils.widgets.text("vector_search_endpoint", "medical_chatbot_endpoint")

# COMMAND ----------

# DBTITLE 1,Load configuration
import os

config = {
    "pdf_path": dbutils.widgets.get("pdf_path"),
    "model_path": dbutils.widgets.get("model_path"),
    "catalog_name": dbutils.widgets.get("catalog_name"),
    "schema_name": dbutils.widgets.get("schema_name"),
    "index_name": dbutils.widgets.get("index_name"),
    "vector_search_endpoint": dbutils.widgets.get("vector_search_endpoint"),
}

for k, v in config.items():
    print(f"{k}: {v}")

# COMMAND ----------

# DBTITLE 1,Verify setup
pdf_exists = os.path.exists(config["pdf_path"])
model_exists = os.path.exists(config["model_path"])

print("=" * 50)
print("Setup Verification")
print("=" * 50)
print(f"\n✓ PDF exists: {pdf_exists}")
if not pdf_exists:
    print(f"  ✗ Update widget pdf_path to your uploaded PDF location.")
    
print(f"\n✓ Model exists: {model_exists}")
if not model_exists:
    print(f"  ✗ Update widget model_path to your saved model location.")

print(f"\n✓ Vector Search Configuration:")
print(f"  - Catalog: {config['catalog_name']}")
print(f"  - Schema: {config['schema_name']}")
print(f"  - Index: {config['index_name']}")
print(f"  - Endpoint: {config['vector_search_endpoint']}")

print("\n" + "=" * 50)
if pdf_exists and model_exists:
    print("✓ Setup complete! Ready to proceed.")
else:
    print("✗ Please fix the issues above before proceeding.")
print("=" * 50)

# COMMAND ----------

