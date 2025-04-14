# 🧠 Mental Health RAG Chatbot (WHO Guidelines)

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that answers mental health questions based on **WHO mental health PDF guidelines**. It uses a combination of **LangChain**, **FAISS**, **Sentence-BERT**, and **Streamlit** to provide accurate, document-grounded answers in real time.

---

## 💡 Key Features

- ✅ **PDF Document Reader**: Extracts and preprocesses WHO mental health PDFs.
- 🔍 **Chunk-based Retrieval**: Splits content into manageable chunks using LangChain.
- 🧠 **Semantic Search**: Embeds chunks using `all-MiniLM-L6-v2` and stores them in a FAISS index.
- 🧾 **Question Answering**: Uses `FLAN-T5` to answer user queries using top relevant chunks.
- 📚 **Source Transparency**: Displays the chunks used to generate each answer.
- 🌐 **Interactive UI**: Built with Streamlit for easy web-based access.

---

## 🗂️ Project Structure

Health_chatbot/ ├── main.py  

## 🚀 Getting Started

### 🔧 1. Clone the Repo

git clone https://github.com/yourusername/mental-health-rag-chatbot.git
cd mental-health-rag-chatbot
📦 2. Install Dependencies
Create a virtual environment and install the required packages:

pip install -r requirements.txt
Or install manually:
pip install streamlit langchain faiss-cpu sentence-transformers transformers
📂 3. Add WHO PDFs
Place your WHO guidelines (e.g., mhGAP, IPT, etc.) into:

data/your_pdfs/
▶️ 4. Run the Chatbot

streamlit run main.py
🧪 Example Questions
What are the WHO-recommended treatments for depression in low-resource settings?

How can caregivers support someone with psychosis?

Are antidepressants safe during breastfeeding?

What are the symptoms of severe depression?

📚 Data Source
This chatbot is grounded on official WHO mental health guidelines, extracted from WHO-provided PDF manuals.

🧠 Model Details
Embeddings: sentence-transformers/all-MiniLM-L6-v2

LLM: google/flan-t5-xl via HuggingFace pipeline

Framework: LangChain + FAISS + Streamlit

📌 Limitations
❌ Does not browse the web or use external APIs.

❗ Answers are only as accurate as the content of the PDFs you load.

⚠️ Not a replacement for professional medical advice.

🤝 Contributions
Feel free to fork the project and suggest improvements via pull requests!

📜 License
This project is open-source under the MIT License.

👤 Author
Zaid Khan
MSc in Natural Language Processing – Cardiff University
This project was built as part of my hands-on learning in Retrieval-Augmented Generation.
