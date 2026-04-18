import os
import streamlit as st
import torch
from typing import List, Any, Optional

from databricks.vector_search.client import VectorSearchClient

# ✅ Updated imports — use langchain_huggingface (the old langchain_community
#    paths emit DeprecationWarning and will be removed).
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# ✅ BaseRetriever and Document now live in langchain_core, not langchain.schema.
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


st.set_page_config(page_title="Medical Chatbot", page_icon="💬", layout="wide")
st.title("Medical Chatbot")
st.caption("Databricks RAG chatbot with Vector Search + phi-2")


class VectorSearchRetriever(BaseRetriever):
    """Custom retriever for Databricks Vector Search.

    BaseRetriever is a Pydantic v2 model in modern LangChain, so fields must
    be declared as class attributes — they can't be set only in __init__.
    """

    # Pydantic fields
    index: Any = None
    embeddings: Any = None
    k: int = 2

    # Allow non-pydantic objects (the Databricks index + HF embeddings)
    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.similarity_search(
            query_vector=query_embedding,
            columns=["id", "text"],
            num_results=self.k,
        )
        docs: List[Document] = []
        for row in results["result"]["data_array"]:
            docs.append(Document(page_content=row[1], metadata={"id": row[0]}))
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


@st.cache_resource
def initialize_qa(
    model_path: str,
    catalog_name: str,
    schema_name: str,
    index_name: str,
    vector_search_endpoint: str,
):
    """Initialize the QA chain with Vector Search and phi-2."""
    prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

    # Initialize embeddings and Vector Search
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name=vector_search_endpoint,
        index_name=f"{catalog_name}.{schema_name}.{index_name}",
    )

    # ✅ Instantiate via keyword args (Pydantic model), not positional.
    retriever = VectorSearchRetriever(index=index, embeddings=embeddings, k=2)

    # Copy model from Volume to local storage (Databricks Apps can't access Volumes directly)
    import shutil
    local_model_path = '/tmp/phi2_model'
    
    if not os.path.exists(local_model_path):
        if os.path.exists(model_path):
            st.info(f"Copying model from Volume ({model_path}) to local storage...")
            try:
                shutil.copytree(model_path, local_model_path)
                st.success(f"✅ Model copied to {local_model_path}")
            except Exception as e:
                st.error(f"Failed to copy model: {e}")
                raise
        else:
            st.error(f"Model not found at {model_path}. Please run 01_setup_environment notebook first.")
            raise FileNotFoundError(f"Model directory not found: {model_path}")
    else:
        st.info(f"Using cached model from {local_model_path}")
    
    # Use the local model path
    actual_model_path = local_model_path
    
    # Load phi-2 model
    tokenizer = AutoTokenizer.from_pretrained(actual_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        actual_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,  # ✅ don't echo the prompt back into the answer
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


with st.sidebar:
    st.header("Configuration")
    model_path = st.text_input(
        "Model Path",
        value=os.getenv(
            "MODEL_PATH",
            "/Volumes/medical_chatbot/medical_data/medical_data/model/",
        ),
    )
    catalog_name = st.text_input(
        "Catalog Name", value=os.getenv("CATALOG_NAME", "medical_chatbot")
    )
    schema_name = st.text_input(
        "Schema Name", value=os.getenv("SCHEMA_NAME", "medical_data")
    )
    index_name = st.text_input(
        "Index Name", value=os.getenv("INDEX_NAME", "medical_bot_index")
    )
    vector_search_endpoint = st.text_input(
        "Vector Search Endpoint",
        value=os.getenv("VECTOR_SEARCH_ENDPOINT", "medical_chatbot_endpoint"),
    )


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


ready = all(
    [model_path, catalog_name, schema_name, index_name, vector_search_endpoint]
)
if not ready:
    st.warning("Fill model path and Vector Search configuration in sidebar to start chat.")
else:
    if not os.path.exists(model_path):
        st.error(f"Model directory not found: {model_path}")
    else:
        qa_chain = initialize_qa(
            model_path, catalog_name, schema_name, index_name, vector_search_endpoint
        )

        for entry in st.session_state.chat_history:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])

        user_prompt = st.chat_input("Ask a medical question")
        if user_prompt:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_prompt}
            )
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = qa_chain.invoke(user_prompt)
                    st.markdown(response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )