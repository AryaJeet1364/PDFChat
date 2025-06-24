# 📄 PDFChatBot – Chat with PDFs using Mistral-7B + LangChain

I built an AI chatbot that lets users **upload and interact with research PDFs** using a custom **RAG (Retrieval-Augmented Generation)** pipeline powered by **LangChain** and the **Mistral-7B** model.

The chatbot allows users to ask questions about their documents, receive context-aware answers, and explore research papers more effectively.

⚠️ **Note**: This is an open-source prototype, and responses may not be fully accurate or suitable for critical research decisions.

Built with:
- `Mistral-7B-Instruct-v0.3` as the LLM
- `FAISS` for semantic retrieval
- `LangChain` for chaining
- `Gradio` for the UI

## 🚀 Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/Aryajeet/PDFChatBot)

## 🖼️ ScreenShot

![alt text](<WhatsApp Image 2025-06-24 at 18.18.44_1e1c965e.jpg>)

---

## 🧠 How It Works

1. **Upload a PDF** – A research paper or document.
2. **Text Extraction & Chunking** – Using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
3. **Embedding** – Converts text into vectors using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Storage** – Vectors are stored in-memory using `FAISS`.
5. **Chat** – Questions are answered by retrieving relevant chunks and generating responses with `Mistral-7B`.

---
