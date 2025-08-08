# voice_bot.py
import streamlit as st
import whisper
import soundfile as sf
import tempfile
import numpy as np
from kokoro import KPipeline
import resampy
import time

def resample_audio(input_wav_path, target_sr=16000):
    data, samplerate = sf.read(input_wav_path)
    if samplerate != target_sr:
        data_16k = resampy.resample(data.T if data.ndim > 1 else data, samplerate, target_sr)
        data_16k = data_16k.T if data.ndim > 1 else data_16k
        sf.write(input_wav_path, data_16k, target_sr)
    return input_wav_path

def transcribe_audio(model, audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav.write(audio_bytes.read() if hasattr(audio_bytes, "read") else audio_bytes)
        temp_wav.flush()
        resample_audio(temp_wav.name)
        result = model.transcribe(temp_wav.name)
    return result["text"]

def generate_tts(tts_pipeline, text):
    generator = tts_pipeline(text, voice='af_sarah', speed=1.0)
    full_audio = np.array([], dtype=np.float32)
    for _, _, audio in generator:
        if full_audio.size == 0:
            full_audio = audio
        else:
            full_audio = np.concatenate((full_audio, audio))
    if isinstance(full_audio, np.ndarray) and full_audio.size > 0:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_f:
            sf.write(tts_f.name, full_audio, 24000)
            return tts_f.name
    return None

def chat_interface(model, tts_pipeline, rag_chain):
    # st.header("🎤 Voice Query")

    # push_to_talk = st.button("Click the mic to speak")
    #
    # if push_to_talk:
    #     st.info("Tap to start speaking, tap again to stop.")

    audio_bytes = st.audio_input("Start speaking here")

    if audio_bytes is not None and rag_chain is not None:
        with st.expander("📝 Transcribing..."):
            with st.spinner("Transcribing audio with Whisper..."):
                transcription = transcribe_audio(model, audio_bytes)
                st.success(f"Transcription: {transcription}")

        if transcription.strip():
            with st.expander("📖 Answer from Documents"):
                answer_placeholder = st.empty()
                full_answer = ""

                def stream_answer(text, chunk_size=40, delay=0.05):
                    for i in range(0, len(text), chunk_size):
                        yield text[i:i + chunk_size]
                        time.sleep(delay)

                with st.spinner("Searching documents for answer..."):
                    answer = rag_chain.invoke(transcription)

                for chunk in stream_answer(answer):
                    full_answer += chunk
                    answer_placeholder.markdown(full_answer)

            if answer.strip():
                if answer.lower() == "i don't know":
                    answer = "Sorry!, Not enough information is available!"
                with st.spinner("Generating speech response..."):
                    tts_audio_path = generate_tts(tts_pipeline, answer)
                    if tts_audio_path:
                        st.audio(tts_audio_path, format="audio/wav", start_time=0)

