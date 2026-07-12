# 🏫 IITB Insti-Assist (Hostel Edition)

**Final Project for the 4-Week NLP Course**

Hey! This is my final project. It's an AI assistant built using Retrieval-Augmented Generation (RAG) to answer questions specifically about IIT Bombay's Hall of Residence Rules. 

Standard LLMs (like ChatGPT) hallucinate a lot when you ask them about IITB-specific stuff, like wDAC rules or visitor timings, because they just guess based on generic university policies. This app fixes that by grounding the LLM in the actual official rulebook using vector search.

### Tech Stack
* **Frontend:** Streamlit
* **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2)
* **Vector DB:** FAISS (running locally)
* **LLM:** Google Gemini 1.5 Flash API

### How to run this thing locally
1. Clone the repo to your machine.
2. Install the requirements (I recommend using a virtual environment): 
   `pip install -r requirements.txt`
3. Run the Streamlit app: 
   `streamlit run app.py`
4. You'll need a free Gemini API key to paste into the sidebar.
5. Upload the `hostel_rules.txt` file (included in this repo) in the sidebar and hit "Process Document".

### Stretch Goals Completed
* **Live File Upload:** Instead of hardcoding the text into the script, you can upload the text file dynamically in the UI.
* **Citation Highlighting:** I added a dropdown expander under the AI's response that shows the exact chunks of text it pulled from FAISS to prove it's not hallucinating.
