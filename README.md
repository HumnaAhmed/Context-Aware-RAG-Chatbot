# 🤖 Context-Aware Chatbot Using LangChain + RAG

## Task Objective
Build a **conversational chatbot** that can remember chat history and retrieve accurate information from custom documents using **Retrieval-Augmented Generation (RAG)**.  

The goal is to create an intelligent chatbot that provides context-aware answers from a knowledge base (documents) instead of relying only on the LLM's internal knowledge.

---

## Dataset
**Custom Knowledge Base**  
File used: `data.txt`

This file can contain any text data such as:
- Wikipedia articles
- Company documents
- Research papers
- Product manuals
- Educational content

---

## Tools & Technologies Used
- **Python**
- **LangChain** (RAG Framework)
- **FAISS** (Vector Database)
- **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)
- **FLAN-T5-base** (Local LLM)
- **Streamlit** (Interactive UI)
- **Transformers** & **Sentence-Transformers**

---

## Steps Performed
1. Loaded custom document (`data.txt`) using TextLoader.
2. Split the document into smaller chunks using CharacterTextSplitter.
3. Created embeddings using Sentence Transformers.
4. Built a FAISS vector store for efficient similarity search.
5. Loaded local LLM (FLAN-T5-base) without using any API key.
6. Designed a custom prompt template for RAG.
7. Built a Retrieval-Augmented Generation (RAG) chain.
8. Implemented **conversational memory** to remember chat history.
9. Developed an interactive UI using Streamlit.
10. Added real-time user input and response display.
11. Included option to clear chat history.

---

## Model & Approach
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `google/flan-t5-base` (Fully Local)
- **Vector Store**: FAISS
- **Chain Type**: `Stuff Documents Chain`
- **Key Features**: 
  - Retrieval-Augmented Generation (RAG)
  - Conversational Memory (Chat History)
  - No API Key Required

---

## Key Results and Findings
- The chatbot successfully retrieves relevant information from the provided document and gives accurate answers.
- Conversational memory allows natural multi-turn conversations.
- Fully runs locally on CPU (no internet or paid API needed after initial setup).
- Good performance on domain-specific questions when relevant content is present in `data.txt`.

---

## Insights
- **RAG** significantly improves answer quality by grounding responses in real documents.
- Local LLMs like FLAN-T5 combined with vector search offer a powerful, private, and cost-free solution.
- Document chunking and embedding quality play a major role in retrieval accuracy.
- This approach is highly useful for building **company chatbots**, **knowledge assistants**, and **document-based Q&A systems**.

---

## Project Files
- `app.py` – Complete Streamlit application with RAG pipeline
- `data.txt` – Custom knowledge base / document
- `requirements.txt` – All dependencies
- `README.md` – Project documentation

---

# 📊 Output & Results

## 1️⃣ Streamlit Chatbot Interface
Interactive web interface where users can chat with the RAG-powered bot.

## 2️⃣ Real-time Document Retrieval
The bot retrieves relevant chunks from `data.txt` before generating answers.

## 3️⃣ Conversational Memory
The chatbot remembers previous messages in the conversation for better context.

## 4️⃣ Example Outputs
- **User**: What is Artificial Intelligence?  
  **Bot**: Artificial Intelligence (AI) is the simulation of human intelligence...

- **User**: What are its main applications?  
  **Bot**: (Answers using previous context + document)

<img width="959" height="469" alt="image" src="https://github.com/user-attachments/assets/b45d21b7-86b4-4e61-8aa2-ff8822997aa3" />

---

## How to Run the Project

```bash
# 1. Clone the repository
git clone <your-repo-link>

# 2. Go to project folder
cd Task-4-Context-Aware-Chatbot

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the chatbot
streamlit run app.py
