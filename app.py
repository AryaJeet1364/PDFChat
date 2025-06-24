import os
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint

# Set token for Hugging Face Inference API
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    max_new_tokens=128,
    temperature=0.5,
)

# Step 1: Upload + Process PDF
def process_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file.name)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    vectordb = FAISS.from_documents(chunks, embeddings)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory
    )

    return qa_chain, "✅ PDF processed! You can now chat.", gr.update(interactive=True), gr.update(interactive=True)

# Step 2: Chatbot logic
def chat_fn(message, chat_history, qa_chain):
    if qa_chain is None:
        return chat_history + [[message, "⚠️ Please upload a PDF first."]], chat_history, qa_chain
    answer = qa_chain.run(message)
    chat_history.append((message, answer))
    return chat_history, chat_history, qa_chain

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Chat with any PDF using Mistral 7B + LangChain")

    qa_chain_state = gr.State(None)
    chat_state = gr.State([])

    upload = gr.File(label="Upload a PDF", file_types=[".pdf"])
    status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot(label="Chat")
    message = gr.Textbox(label="Ask a question", placeholder="Type your question...", interactive=False)
    send_btn = gr.Button("Send", interactive=False)

    upload.change(fn=process_pdf, inputs=upload, outputs=[qa_chain_state, status, message, send_btn])
    send_btn.click(fn=chat_fn, inputs=[message, chat_state, qa_chain_state], outputs=[chatbot, chat_state, qa_chain_state])

demo.launch(share=True)