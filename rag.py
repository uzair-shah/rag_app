from pypdf import PdfReader
import re
from pprint import pprint
from sentence_transformers import SentenceTransformer
import numpy as np
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
    pprint(text_list)
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
            print(f"Printed page number {page.page_number} and added {len(page_text)} characters")
            full_text+=" \n" + page_text
    return full_text


def chunk_text(text, chunk_size, overlap):
    text_list = []
    i = 0
    while i < len(text):
        print(f'Start {i}, End {i + chunk_size}')
        text_list.append( text[i:i+chunk_size])
        i = i + (chunk_size - overlap)
    return text_list

def embed_chunks(chunks):
    #Loading a pretrained Sentence Transformer model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # 2. Calculate embeddings by calling model.encode()
    embeddings = model.encode(chunks)
    print(embeddings.shape)
    # print(embeddings)
    chunk_embs = [{"text": chunk, "embeddings": embedding} for chunk,embedding in zip(chunks,embeddings)]
    return chunk_embs

def search(question, chunk_embs, model, k=5):
    
    q_embedding = model.encode(question)
    # print(q_embedding.shape)
    embeddings = []
    
    for item in chunk_embs:
        embeddings.append(item['embeddings'])
    
    sim = model.similarity(q_embedding,embeddings) #gives similarity scores
    
    indices = np.argsort(-sim) #finding the arrays with highest similarity values
    
    return [{'text':chunk_embs[i]['text'], 'score': sim[0,i].item()} for i in indices[0,:k]]




file_name = "Warrington-JugglingProbabilities-2005.pdf"
text = pdf_to_text(file_name)
cleaned_text = clean_text(text)
chunk_embs = chunk_text(cleaned_text,200,50)

#shape of first embedding
print(chunk_embs[0]['embeddings'].shape)
print(chunk_embs[0]['embeddings'][:5])
#printing embedding values
for item in chunk_embs:
    if 'Markov chain' in item['text']:
        numbers = item['embeddings'][:5]
        print(numbers)
        break

#testing the results of search function
for q in [
    "What is a Markov chain?",
    "Who is the author?",
    "How does juggling work?",
    "Banana sandwich recipe",
]:
    print(f"\n=== {q} ===")
    results = search(q, chunk_embs, model, k=3)
    for r in results:
        print(f"  ----{r['score']:.3f}--- | {r['text'][:100]}")