from pypdf import PdfReader
import re
from pprint import pprint
from sentence_transformers import SentenceTransformer
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
import ollama
from ollama import chat
import time
import json 

load_dotenv()

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
    '''returns top k answers to question based on similarity score'''
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
    context = "\n---\n".join(text_list)
    return f"Context:\n{context}\n\nQuestion: {question}"
    

def ask_llm_gemini(question, retrieved_chunks, client, max_retries=3):
    prompt = build_prompt(question, retrieved_chunks)
    
    for attempt in range(max_retries):
        time.sleep(20)
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful assistant. Use only the provided context to answer the user's question. If the context does not contain the answer, say so honestly."
                ),
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # last attempt — let it crash
            wait_time = 5 ** attempt  # 1s, 25s, 125s — exponential backoff a^x
            print(f"  API error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

def ask_llm_deep_seek(question, retrieved_chunks,model='deepseek-r1'):
    prompt = build_prompt(question, retrieved_chunks)
    
    response = chat(
            model=model,
            messages=[{'role': 'system', 'content': 'You are a helpful assistant. Use only the provided context'},
                {'role': 'user', 'content': prompt}],
        )
    return response.message.content

def score_query(test_query, answer):
    answer_lower = answer.lower()
    
    if test_query["should_answer"]:
        # Positive test: look for any expected keyword in the answer
        matches = [kw for kw in test_query["expected_keywords"] 
                   if kw.lower() in answer_lower]
        passed = len(matches) > 0
        if passed:
            reason = f"found keywords: {matches}"
        else:
            reason = f"missing all expected keywords: {test_query['expected_keywords']}"
    else:
        # Negative test: look for any refusal phrase in the answer
        refusals = [kw for kw in test_query["refusal_keywords"] 
                    if kw.lower() in answer_lower]
        passed = len(refusals) > 0
        if passed:
            reason = f"refused with: {refusals}"
        else:
            reason = "did not refuse"
    
    return {
        "question": test_query["question"],
        "category": test_query["category"],
        "answer": answer,
        "passed": passed,
        "reason": reason,
    }

def run_eval(test_queries, chunk_embs, model, client, agent='gemini'):
    """Run all test queries through the pipeline. Return list of result dicts."""
    # For each query: call search → ask_llm → score_query → collect
    eval_results = []
    for query in test_queries:
    
        question = query['question']
        results = search(question, chunk_embs, model, k=3)
        start_time = time.perf_counter()
        
        if agent == 'gemini':
            answer = ask_llm_gemini(question, results,client)
        else:
            answer = ask_llm_deep_seek(question, results)
        end_time = time.perf_counter()
        
        if agent == 'gemini':
            elapsed_time = end_time - start_time - 20 #account for sleep time in gemini call
        else:
            elapsed_time = end_time - start_time #time taken to 
        print(f"LLM response time: {elapsed_time:.4f} seconds")

        score_result = score_query(query,answer)
        eval_results.append(score_result)
        print(f"[{len(eval_results)}/{len(test_queries)}] {'Correct' if score_result['passed'] else 'Incorrect'} {question}")
    return eval_results

def print_summary(results):
    """Pretty-print the results: per-query + overall + per-category."""
    for result in results:
        pprint(f"Question: {result['question']}")
        pprint(f"Category: {result['category']}")
        pprint(f"Passed: {result['passed']}")
        pprint(f"Answer: {result['answer']}")

#initialising 
file_name = "Warrington-JugglingProbabilities-2005.pdf"
text = pdf_to_text(file_name)
cleaned_text = clean_text(text)
chunk_embs = embed_chunks(chunk_text(cleaned_text, 800, 100),model)


with open('test_queries.txt', 'r', encoding='utf-8') as f:
    test_queries = json.load(f)

client = genai.Client()  #loading gemini 
eval_results_gemini = run_eval(test_queries, chunk_embs, model, client)
eval_results_deepseek = run_eval(test_queries, chunk_embs, model, client, agent = 'deep_seek')
print_summary(eval_results_deepseek)

# while True:
#     q = input("\nAsk a question (or 'quit'): ").strip()
#     if q.lower() == "quit":
#         break
#     if not q:
#         continue
#     results = search(q, chunk_embs, model, k=5)
#     answer = ask_llm_gemini(q, results,client)
#     print("\n" + answer)



