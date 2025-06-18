import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

import os
openai_api_key = os.getenv("OPENAI_API_KEY")

# ---------------- Streamlit UI ----------------
st.title("🎥 YouTube Video QA with LangChain + OpenAI")

video_id = st.text_input("🔗 Enter YouTube Video ID", placeholder="e.g. 1YnNV1fLuww")
llm_model = st.selectbox("🧠 Choose OpenAI Model", ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"])
temperature = st.slider("🌡️ LLM Temperature", 0.0, 1.0, 0.2, 0.05)
user_question = st.text_input("❓ Ask a question from the transcript")

if st.button("Run RAG Pipeline"):
    if not video_id:
        st.warning("Please enter a valid YouTube video ID.")
    else:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
        except TranscriptsDisabled:
            st.error("⚠️ No captions available for this video.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()

        # ---------------- RAG Pipeline ----------------
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=openai_api_key)
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
            """,
            input_variables=['context', 'question']
        )

        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        parser = StrOutputParser()
        llm = ChatOpenAI(model=llm_model, temperature=temperature)
        main_chain = parallel_chain | prompt | llm | parser

        # ---------------- Output ----------------
        with st.spinner("🔍 Processing..."):
            result = main_chain.invoke(user_question)
            st.subheader("📌 Answer:")
            st.write(result)
