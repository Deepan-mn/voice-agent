# main.py
import streamlit as st
import rag
import voice_bot

# Initialize/load global models here (or move to voice_bot.py and rag.py if preferred)
if "model" not in st.session_state:
    import whisper
    st.session_state.model = whisper.load_model("base")

if "tts_pipeline" not in st.session_state:
    from kokoro import KPipeline
    st.session_state.tts_pipeline = KPipeline(lang_code='a')

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

model = st.session_state.model
tts_pipeline = st.session_state.tts_pipeline
rag_chain = st.session_state.rag_chain

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align:center; color:#4B8BBE;'>🎙️ Voice Agent</h1>", unsafe_allow_html=True)
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #4B8BBE;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        border: none;
        font-size: 18px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #306998;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# Sidebar - Knowledge Base Upload and Processing
with st.sidebar.expander("📚 Knowledge Base Upload", expanded=True):
    files = st.file_uploader("Upload PDFs or Text files", type=['pdf', 'txt'], accept_multiple_files=True)
    if st.button("Process Knowledge Base"):
        if not files:
            st.warning("Upload files to process first!")
        else:
            docs = rag.load_documents(files)
            if docs:
                rag_chain_obj, retriever = rag.build_rag_chain(docs)
                st.session_state.rag_chain = rag_chain_obj
                st.session_state.vectorstore_retriever = retriever
                st.success("Knowledge base processed and ready! Use the voice query in the main area.")
            else:
                st.error("No documents could be loaded from the files.")

# Main area - Voice Query Chat Bot
voice_bot.chat_interface(model, tts_pipeline, st.session_state.rag_chain)
