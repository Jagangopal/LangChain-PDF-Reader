import hashlib
import os

# Task 1: Import the required libraries
import streamlit as st
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="PDF Reader", page_icon="📄")
st.title("📄 Chat with your PDF")


# Task 2: Set up the Groq API key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Groq API key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Your key is only used for this session. You can also set GROQ_API_KEY in a .env file.",
    )

if not api_key:
    st.info("Enter your Groq API key in the sidebar to get started.")
    st.stop()


# Task 3: Upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is None:
    st.info("Upload a PDF to start asking questions about it.")
    st.stop()


# Task 4: Extract the text from the PDF
def extract_text(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# Task 5: Split the text into chunks
def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


# Task 6: Generate embeddings for each chunk and create a searchable database
# Groq has no embeddings API, so embeddings are computed locally with fastembed.
class LocalEmbeddings(Embeddings):
    def __init__(self, model: TextEmbedding):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embeddings() -> LocalEmbeddings:
    return LocalEmbeddings(TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=".fastembed_cache"))


def build_vector_store(chunks: list[str]) -> InMemoryVectorStore:
    return InMemoryVectorStore.from_texts(chunks, load_embeddings())


# Only rebuild the database when a different file is uploaded
file_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(file_bytes).hexdigest()

if st.session_state.get("file_hash") != file_hash:
    with st.spinner("Reading the PDF and building the search index..."):
        text = extract_text(uploaded_file)
        if not text.strip():
            st.error("No text could be extracted from this PDF (it may be a scanned image).")
            st.stop()
        chunks = split_text(text)
        st.session_state.vector_store = build_vector_store(chunks)
        st.session_state.file_hash = file_hash
        st.session_state.chat_history = []
    st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}.")


# Task 7: Initialize and display the chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)


# Task 8: Answer the question using the most relevant chunks and the chat history
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions about a PDF document. "
            "Use only the context below and the conversation so far. "
            "If the answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0, api_key=api_key)
chain = prompt | llm | StrOutputParser()
retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})

question = st.chat_input("Ask a question about the PDF")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    with st.chat_message("assistant"):
        answer = st.write_stream(
            chain.stream(
                {
                    "context": context,
                    "chat_history": st.session_state.chat_history,
                    "question": question,
                }
            )
        )
        with st.expander("Sources"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}**")
                st.text(doc.page_content)

    # Task 9: Append the question and answer to the chat history
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))
