# ============================================================
# TASK 4: Context-Aware RAG Chatbot (FINAL STABLE VERSION)
# ✅ tf-keras fix | Pure CPU | Conversational Memory
# ============================================================

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Transformers imports
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

# ------------------------------------------------------------
# 1. Load Document
# ------------------------------------------------------------
loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()
st.sidebar.success("✅ data.txt loaded successfully!")

# ------------------------------------------------------------
# 2. Split into Chunks
# ------------------------------------------------------------
text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# ------------------------------------------------------------
# 3. Embeddings
# ------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------------------------------------
# 4. FAISS Vector Store
# ------------------------------------------------------------
vector_db = FAISS.from_documents(docs, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 4})


# ------------------------------------------------------------
# 5. Load FLAN-T5 (Manual Loading - Stable)
# ------------------------------------------------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"  # Change to "google/flan-t5-small" for faster speed

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=350,
        temperature=0.7,
        do_sample=False,
        top_p=0.9
    )
    return HuggingFacePipeline(pipeline=pipe)


llm = load_llm()

# ------------------------------------------------------------
# 6. Prompt Template
# ------------------------------------------------------------
system_prompt = (
    "You are a helpful, accurate assistant. "
    "Answer the question using ONLY the information from the context below. "
    "If the answer is not in the context, reply with 'I don't know based on the provided document.'\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# ------------------------------------------------------------
# 7. RAG Chain
# ------------------------------------------------------------
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)

# ------------------------------------------------------------
# 8. Streamlit UI + Memory
# ------------------------------------------------------------
st.title("🤖 Context-Aware RAG Chatbot")
st.caption("Local • No API Key • Remembers Conversation")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask any question from your document..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add history for better context
    history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}"
                              for m in st.session_state.messages[:-1]])

    full_prompt = f"Previous conversation:\n{history_text}\n\nCurrent Question: {user_input}" if history_text else user_input

    with st.spinner("🤔 Thinking using RAG..."):
        result = qa_chain.invoke({"input": full_prompt})
        answer = result.get("answer", "Sorry, I could not find the answer in the document.")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# Clear button
if st.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.info("📁 data.txt file same folder mein honi chahiye")