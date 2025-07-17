
from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
# import docx
# import spire.doc
import docx2txt2
from bs4 import BeautifulSoup
from markdown import markdown
import re

def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    return text


def clean_text_from_raw_pdf(rawtext):
    # Remove extra whitespace (multiple spaces, newlines, tabs)
    cleaned_text = re.sub(r'\s+', ' ', rawtext)
    # Remove special characters or symbols (customize as needed)
    # This example keeps alphanumeric characters, spaces, periods, and commas
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,/@]', ' ', cleaned_text)
    # Remove leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def main():
    # print("This is the main function of app.py")
    # load_dotenv()
    # print(f"Just to test the .env variables work, \nhere is the content of my DUMMY_KEY: {os.getenv('DUMMY_KEY')}")

    st.set_page_config(
        page_title="ResRAG App."
    )
    st.header("ResRAG App!")

    uploaded_files = st.file_uploader("Please upload your resume files here!", accept_multiple_files=True)

    for uploaded_file in uploaded_files:
        # st.write("filename:", uploaded_file.name)

        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            print(f"/n/n PDF TEXT BEFORE CLEANING: {text}")
            text = clean_text_from_raw_pdf(text)
            st.text_area(f"Content of Resume file {uploaded_file.name}", value=text, height=700)

        elif uploaded_file.name.endswith('.txt'):
            text = ""
            text = uploaded_file.read().decode("utf-8")
            st.text_area(f"Content of Resume file {uploaded_file.name}", value=text, height=700)

        elif uploaded_file.name.endswith('.docx'):
            # doc = docx.Document(uploaded_file)
            # text = "\n".join([para.text for para in doc.paragraphs])
            # st.text_area(f"Content of Resume file {uploaded_file.name}", value=text, height=700)

            # full_text = []
            # document = docx.Document(uploaded_file)
            # for paragraph in document.paragraphs:
            #     full_text.append(paragraph.text)
            # st.text_area(f"Content of Resume file {uploaded_file.name}", value=full_text, height=700)
            # print(f"/n/n {full_text} ")

            # # Create a Document object
            # document = spire.doc.Document()
            # # Load a Word document
            # document.LoadFromFile("Sample.docx")

            # # Extract the text of the document
            # document_text = document.GetText()

            # st.text_area(f"Content of Resume file {uploaded_file.name}", value=document_text, height=700)
            # print(f"/n/n {document_text} ")
            # document.Close()

            document_text = docx2txt2.extract_text(uploaded_file)
            st.text_area(f"Content of Resume file {uploaded_file.name}", value=document_text, height=700)
            

        elif uploaded_file.name.endswith('.md'):
            text = uploaded_file.read().decode("utf-8")
            text = markdown_to_text(text)
            st.text_area(f"Content of Resume file {uploaded_file.name}", value=text, height=700)

        else:
            st.error(f"Unsupported file type: {uploaded_file.name}. Please upload a PDF, TXT, DOCX, or MD file.")





if __name__ == "__main__":
    main()