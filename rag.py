from pypdf import PdfReader
import re
from pprint import pprint

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
        elif not line.strip():  # empty or only whitespace
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


file_name = "Warrington-JugglingProbabilities-2005.pdf"
text = pdf_to_text(file_name)
cleaned_text = clean_text(text)