# Advanced RAG Chatbot

## Features
- Document Q&A with source citation
- Document summarization mode
- Cohere reranking for improved relevance
- Adjustable chunking parameters
- Conversation history
- Support for PDF, TXT, and other document types

## Setup
1. Clone repository:
   ```bash
   git clone https://github.com/Edwin420s/naive_rag_chatbot.git
   cd rag-pro
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add API keys to `.env`:
   ```env
   OPENAI_API_KEY=your_openai_key
   COHERE_API_KEY=your_cohere_key  # For reranking
   ```
4. Add documents to `docs/` folder
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Example Queries
- "What are the key findings in the research document?"
- "Summarize the main points of the contract"
- "Explain the methodology used in the report"

## Deployment
Deploy to Streamlit Cloud:
1. Create `requirements.txt` with dependencies
2. Push to GitHub repository
3. Go to [Streamlit Cloud](https://streamlit.io/cloud) and import repository

## Limitations
- Large documents (>100 pages) may require more processing time
- Accuracy depends on document quality and specificity of questions