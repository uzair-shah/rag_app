from pypdf import PdfReader
import re
from pprint import pprint
from sentence_transformers import SentenceTransformer
import numpy as np
from google import genai
from google.genai import types
import os 
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

#Loading a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def clean_text(full_text):
    #removes boilerplate 
    full_text = re.sub(r"This content.*?/terms", "", full_text, flags=re.DOTALL)
    full_text = re.sub(r"THE MATHE.*?ERICA", "", full_text, flags=re.DOTALL)
    full_text = re.sub(r"JUGGLING.*?2005", "", full_text, flags=re.DOTALL)
    text_list = []
    for line in full_text.split("\n"):
        if "\x00" in line: #removes nullspace
            line = line.replace("\x00",'')
            text_list.append(line)
        elif not line.strip():  # empty or only whitespace on line
            continue
        else: #adds items back to list
            text_list.append(line)
    
    cleaned_text = "\n".join(text_list)
    return cleaned_text

def pdf_to_text(file_name):
    '''Takes in file name and processes it as a text string'''
    full_text = ''
    with open(file_name,"rb") as file:
        reader = PdfReader(file)
        number_of_pages = len(reader.pages)
        # Loop through all pages
        for page in reader.pages: #page is an object of pageobject class
            page_text = page.extract_text()
            full_text+=" \n" + page_text
    return full_text


def chunk_text(text, chunk_size, overlap):
    text_list = []
    i = 0
    #chunking text with overlap to maintain context
    while i < len(text):
        text_list.append( text[i:i+chunk_size])
        i = i + (chunk_size - overlap)
    return text_list

def embed_chunks(chunks,model):
    '''Convert text to vector embeddings'''
    # Calculate embeddings by calling model.encode()
    embeddings = model.encode(chunks)
    #Store text-embedding pairs 
    chunk_embs = [{"text": chunk, "embeddings": embedding} for chunk,embedding in zip(chunks,embeddings)]
    return chunk_embs


def search(question, chunk_embs, model, k=5):
    #embedding question to a vector
    q_embedding = model.encode(question)
    #storing embeddings separately
    embeddings = np.array([item['embeddings'] for item in chunk_embs]) #stacks into 2d array
    sim = model.similarity(q_embedding,embeddings) #gives similarity scores
    indices = np.argsort(-sim) #finding the arrays with highest similarity values
    return [{'text':chunk_embs[i]['text'], 'score': sim[0,i].item()} for i in indices[0,:k]]

def build_prompt(question, retrieved_chunks):
    '''Pulls out text from retrieved chunks and applies context and question headers to final string'''
    text_list = [item['text'] for item in retrieved_chunks]
    return 'Context: ' + '---\n---'.join(text_list) + '---\n---' 'Question: ' + question

def ask_llm(question, retrieved_chunks,client):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant. Use only the provided context to answer the user's question. If the context does not contain the answer, say so honestly"),
        contents=build_prompt(question, retrieved_chunks)
    )
    return response.text

#initialising 
file_name = "Warrington-JugglingProbabilities-2005.pdf"
text = pdf_to_text(file_name)
cleaned_text = clean_text(text)
chunk_embs = embed_chunks(chunk_text(cleaned_text, 800, 100),model)

client = genai.Client()  # once, outside the loop

while True:
    q = input("\nAsk a question (or 'quit'): ").strip()
    if q.lower() == "quit":
        break
    if not q:
        continue
    results = search(q, chunk_embs, model, k=5)
    answer = ask_llm(q, results,client)
    print("\n" + answer)