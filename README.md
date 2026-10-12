# LangChain PDF Reader

An interactive PDF reader built with LangChain and Streamlit. Upload a PDF, and the app splits its text into chunks, embeds each chunk locally with fastembed, and stores them in an in-memory vector database. Ask questions in the chat: the app finds the most relevant chunks and sends them, along with the chat history, to an LLM to generate an answer.

> This project uses your own Groq API key (create one at https://console.groq.com/keys). Embeddings run locally, so no other key is needed. The embedding model (~130 MB) downloads on first use.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Optionally copy `.env.example` to `.env` and add your key. Otherwise, enter it in the app's sidebar.

## Run

```bash
streamlit run app.py
```

Press `Ctrl + C` in the terminal to stop the app; run the command above again to restart it.

## How it works

Each step is marked with a `# Task N` comment in [app.py](app.py):

1. Import libraries
2. Set up the Groq API key
3. Upload a PDF
4. Extract the text (`pypdf`)
5. Split it into overlapping chunks (`RecursiveCharacterTextSplitter`)
6. Embed the chunks and build an in-memory vector store (local `BAAI/bge-small-en-v1.5` via fastembed)
7. Initialize and display the chat history
8. Retrieve the top 3 chunks for the question and answer with Groq (`openai/gpt-oss-120b` by default; set `GROQ_MODEL` in `.env` to use another [Groq model](https://console.groq.com/docs/models))
9. Append the question and answer to the chat history
