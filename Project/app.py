import streamlit as st
import google.generativeai as genai
from rag_engine import RAGPipeline

st.set_page_config(page_title="Insti-Assist", page_icon="🏫")

st.title("IITB Hostel Assistant 🤖")
st.write("Ask me anything about the Hall of Residence rules.")

# Streamlit session state is kinda confusing. Need this so the DB doesn't reset every time you type.
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGPipeline()
    st.session_state.data_loaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    st.caption("Get a free one from Google AI Studio if you don't have it.")
    
    st.divider()
    
    st.header("Upload Rulebook")
    file = st.file_uploader("Upload hostel_rules.txt", type=['txt'])
    
    if st.button("Process Document") and file:
        with st.spinner("Chunking text and building vector database..."):
            text = file.getvalue().decode("utf-8")
            chunks = st.session_state.rag_system.clean_and_chunk(text)
            
            if st.session_state.rag_system.create_vector_db(chunks):
                st.session_state.data_loaded = True
                st.success(f"Done! Loaded {len(chunks)} rules successfully.")
            else:
                st.error("Failed to parse document")

# --- MAIN CHAT ---
# Show previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("E.g., Are we allowed to use induction heaters?")

if user_input:
    # 1. Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Error handling
    if not st.session_state.data_loaded:
        st.error("⚠️ Please upload the rulebook in the sidebar first!")
    elif not api_key:
        st.error("⚠️ API key is missing.")
    else:
        # 2. Search FAISS for relevant rules
        retrieved_rules = st.session_state.rag_system.retrieve(user_input, k=3)
        context_block = "\n\n".join(retrieved_rules)
        
        # 3. Build Prompt (Strict guardrails to stop hallucinations)
        prompt = f"""
        You are a helpful assistant for IIT Bombay students.
        Answer the user's question using ONLY the provided rules below.
        
        Rules Context:
        {context_block}
        
        User Question: {user_input}
        
        CRITICAL INSTRUCTIONS:
        - If the context doesn't contain the answer, say EXACTLY: "I am sorry, but I do not have that information."
        - Do not make assumptions or use external knowledge about other universities.
        - Keep it concise.
        """
        
        # 4. Call Gemini API
        with st.spinner("Searching rulebook..."):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                
                # Show response
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                    
                    # Stretch Goal: Show Citations
                    with st.expander("View Retrieved Context (Sources)"):
                        for idx, rule in enumerate(retrieved_rules):
                            st.write(f"**Chunk {idx+1}:** {rule}")
                            
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            
            except Exception as e:
                st.error(f"API Error: {e}")
