# End-to-End Medical Chatbot Using Llama 2

This project demonstrates how to build a conversational chatbot using the **Llama 2** model. A PDF containing medical information is indexed with **Pinecone** and queried through a web chat interface.

## Features

- **Document ingestion**: `Medical_book.pdf` is split into text chunks and embedded with `sentence-transformers`.
- **Pinecone vector store**: Embeddings are stored and searched using Pinecone.
- **Llama 2 inference**: Responses are generated locally with a quantized Llama 2 model via `ctransformers`.
- **Web chat UI**: A simple interface supports interactive Q&A.

## How to Run

### 1. Clone the Repository

Project repo: `https://github.com/`

### 2. Create and Activate Conda Environment

```bash
conda create -n mchatbot python=3.8 -y
conda activate mchatbot
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add:

```env
PINECONE_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 5. Download the Llama 2 Quantized Model

Model file:

```text
llama-2-7b-chat.ggmlv3.q4_0.bin
```

Download from:

`https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main`

Keep the model file in the `model/` directory.

### 6. Build the Vector Index

```bash
python store_index.py
```

### 7. Start the Application

```bash
python app.py
```

Open `localhost` in your browser.

## Tech Stack

- Python
- LangChain
- Flask
- Meta Llama 2
- Pinecone