## Task 4: Context-Aware Chatbot Using LangChain + RAG

### Objective
Build a conversational chatbot that can remember context and retrieve information from custom documents using Retrieval-Augmented Generation (RAG).

### Dataset Used
- Custom `data.txt` file (can contain Wikipedia pages, company documents, or any knowledge base)

### Technologies & Models Used
- **LangChain** (for RAG pipeline and chains)
- **FAISS** (Vector Database)
- **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)
- **FLAN-T5-base** (Local LLM - No API Key)
- **Streamlit** (for UI/Deployment)
- **CharacterTextSplitter** for chunking

### Key Features
- Document loading and chunking
- Vector embeddings and similarity search
- Retrieval-Augmented Generation (RAG)
- Conversational memory (remembers chat history)
- Fully local (No OpenAI or API keys required)
- Clean Streamlit interface

### How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
Screenshots
(Apne chatbot ke screenshots daal dena yahan)
Key Learnings

RAG Architecture
Document embeddings and vector stores
Conversational AI with memory
Local LLM integration


Made with ❤️ for AI/ML Internship
text### Step 4: requirements.txt File Banao

Project folder mein `requirements.txt` naam ki file banao aur isme yeh likho:

```txt
streamlit
langchain
langchain-community
langchain-core
langchain-text-splitters
langchain-huggingface
langchain-classic
faiss-cpu
sentence-transformers
transformers==4.45.2
tf-keras
torch