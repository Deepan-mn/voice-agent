# voice-agent

# 🎙️ Voice Agent – RAG + Voice Chatbot

This application is a **voice-enabled AI assistant** that lets you **talk to your documents**.  
Upload your PDFs or text files, and simply speak your question – the bot will transcribe your voice, retrieve the most relevant answer from your knowledge base using **RAG (Retrieval-Augmented Generation)**, and reply back in a natural, human-like voice.  

---

## ✨ Features
- 🎤 **Voice Input:** Speak instead of typing – powered by OpenAI Whisper.
- 📚 **Document Knowledge Base:** Upload PDFs or TXT files to build a searchable knowledge base.
- 🔍 **Retrieval-Augmented Generation:** Finds the most relevant info from your documents before answering.
- 🗣️ **Text-to-Speech:** Natural-sounding audio replies using Kokoro TTS.
- ⚡ **Real-time Interaction:** Smooth and quick responses in a friendly chat interface.

---

## 🛠️ Tech Stack
- **Frontend/UI:** [Streamlit](https://streamlit.io/)
- **Speech-to-Text (ASR):** [Whisper](https://github.com/openai/whisper)
- **Text-to-Speech (TTS):** [Kokoro](https://github.com/hexgrad/kokoro)
- **Document Processing & RAG:** LangChain + Vector Stores
- **Backend Language:** Python3.10

---

## 🚀 How It Works
1. **Upload Documents** – PDF or TXT files via the sidebar.
2. **Process Knowledge Base** – Files are chunked, embedded, and stored in a vector database.
3. **Ask via Voice** – Speak your query into the mic.
4. **RAG Retrieval** – Finds and ranks relevant chunks from your uploaded content.
5. **Answer Generation** – Summarizes and formats the best answer.
6. **Voice Response** – Converts the answer into natural speech and plays it.

---

## 📦 Installation
```
git clone https://github.com/yourusername/voice-agent.git
cd voice-agent
pip install -r requirements.txt
```

## ▶️ Run the App
```
streamlit run main.py
```

## 📌 Notes
1. Make sure you have **FFmpeg** installed for Whisper.
2. Supports multiple files and multiple queries in a session.
3. Best used with clear audio for optimal transcription accuracy.
