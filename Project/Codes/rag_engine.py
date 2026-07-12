import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Setting this up as a class because it was getting too messy doing it all in app.py
class RAGPipeline:
    def __init__(self):
        # Using the small MiniLM model because my laptop fan was going crazy with the bigger ones
        print("Loading embedding model... this might take a sec")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []
        
    def clean_and_chunk(self, text):
        # 1. Clean up OCR garbage
        # The pdf parser picked up some random dates and signatures from the scan, removing them here
        text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', text)
        text = re.sub(r'(Abhay|Novayane|thay\.|cuje Novayana)', '', text, flags=re.IGNORECASE)
        
        # 2. Split by the main headings (e.g., "1. General Conduct")
        # Regex looks for a newline, a number, a dot, and a space. (Took me forever to get this to work)
        raw_sections = re.split(r'\n(?=\d\.\s+[A-Z])', text)
        
        processed_chunks = []
        
        # 3. Add context to the chunks so the embeddings actually make sense
        for sec in raw_sections:
            sec = sec.strip()
            if not sec: 
                continue
                
            lines = sec.split('\n', 1) 
            
            # Check if it starts with a number like "1. "
            if re.match(r'^\d\.\s+', lines[0]):
                section_name = lines[0].strip()
                content = lines[1] if len(lines) > 1 else ""
                
                # Split by bullet points
                bullets = content.split('•')
                for b in bullets:
                    # Clean up stray line breaks inside the bullet point
                    b = b.replace('\n', ' ').strip()
                    b = re.sub(r'\s+', ' ', b) 
                    
                    if b:
                        # MAGIC SAUCE: Append the section name to EVERY bullet point
                        # (I realized my searches were returning random rules out of context before I did this)
                        chunk_text = f"[Category: {section_name}] {b}"
                        processed_chunks.append(chunk_text)
                        # print(chunk_text)  # left this here for debugging
            else:
                # Catch the intro/outro paragraphs (like the wDAC warnings)
                body = sec.replace('\n', ' ').strip()
                body = re.sub(r'\s+', ' ', body)
                if body:
                    processed_chunks.append(f"[General Context] {body}")
                    
        return processed_chunks

    def create_vector_db(self, text_chunks):
        if len(text_chunks) == 0:
            return False
            
        self.chunks = text_chunks
        
        # Convert text chunks to vectors
        embeddings = self.model.encode(self.chunks)
        dim = embeddings.shape[1]
        
        # Load into FAISS index for fast searching
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype('float32'))
        return True

    def retrieve(self, query, k=3):
        if not self.index:
            return []
            
        # Encode the user's question
        query_vector = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                results.append(self.chunks[idx])
        return results
