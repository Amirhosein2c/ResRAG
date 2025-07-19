
from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
import docx2txt2
from bs4 import BeautifulSoup
from markdown import markdown
import re
import shutil
from langchain_core.documents import Document
# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama.chat_models import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_ollama.chat_models import ChatOllama


CHROMA_PATH = "./DBPATH"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

PROMPT_TEMPLATE = """

You are an HR assistant that analyzes resumes and explains how well they match job search queries. 
You receive the following context: 



and must evaluate how relevant each resume is to the search requirements.

When given a search query: 

{question}

and resume content: 

{context}

you must:

Analyze how well each resume matches the query requirements
Provide a relevance score (1-10)
Explain why each resume is a good or poor match
Highlight the most relevant qualifications

Response Format
For each matching resume, respond exactly like this:
Resume Match #[number]
Match Score: [1-10]/10
Key Strengths: [List 2-3 most relevant qualifications from the resume]
Match Explanation: [2-3 sentences explaining why this resume matches or doesn't match the query]
Location: [If mentioned in resume]
Experience Level: [Junior/Mid/Senior based on resume content]

Scoring Guidelines

9-10: Excellent match - meets most/all requirements
7-8: Good match - meets key requirements with minor gaps
5-6: Moderate match - some relevant skills but missing important ones
3-4: Weak match - few relevant qualifications
1-2: Poor match - minimal or no relevant qualifications

Important Rules

Only use information actually present in the resume content:

{context}

Be specific about which skills/experiences match the query:

{question}

If location is important in the query, mention it in your analysis
Be honest about gaps or missing qualifications
Focus on technical skills, experience, and education mentioned in both query and resume

Example
Query: "Looking for a data scientist with Python and TensorFlow experience in Germany"
Resume contains: Python programming, machine learning projects, TensorFlow usage, located in Berlin
Your response should be:
Resume Match #1
Match Score: 8/10
Key Strengths: Python programming experience, TensorFlow projects, machine learning background
Match Explanation: Strong match for data scientist role. Resume shows Python programming skills and specific TensorFlow experience through ML projects. Located in Berlin, Germany which meets location requirement.
Location: Berlin, Germany
Experience Level: Mid-level


Pay attention:

Help the user based only on the following context:

{context}

---

Answer the question based on the above context: 

{question}

"""


def markdown_to_text(markdown_string):
    html = markdown(markdown_string)
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    return text


def clean_text_from_raw_pdf(rawtext):
    cleaned_text = re.sub(r'\s+', ' ', rawtext)
    cleaned_text = re.sub(r'[^\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z0-9\s.,/@]', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def oldstyle_reader(uploaded_files):
    extracted_texts = list()
    for uploaded_file in uploaded_files:

        text = ""
        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
            # print(f"/n/n PDF TEXT BEFORE CLEANING: {text}")
            text = clean_text_from_raw_pdf(text)
            # st.text_area(f"Content of Resume file {uploaded_file.name}", value=text, height=700)
            extracted_texts.append(Document(page_content=text, metadata={"source": str(uploaded_file.name)}))


        elif uploaded_file.name.endswith('.txt'):
            text = uploaded_file.read().decode("utf-8")
            # st.text_area(f"Content of Resume file {uploaded_file.name}", value=text, height=700)
            extracted_texts.append(Document(page_content=text, metadata={"source": str(uploaded_file.name)}))

        elif uploaded_file.name.endswith('.docx') :
            text = docx2txt2.extract_text(uploaded_file)
            # st.text_area(f"Content of Resume file {uploaded_file.name}", value=text, height=700)
            extracted_texts.append(Document(page_content=text, metadata={"source": str(uploaded_file.name)}))
            
        elif uploaded_file.name.endswith('.md'):
            text = uploaded_file.read().decode("utf-8")
            text = markdown_to_text(text)
            # st.text_area(f"Content of Resume file {uploaded_file.name}", value=text, height=700)
            extracted_texts.append(Document(page_content=text, metadata={"source": str(uploaded_file.name)}))

        else:
            st.error(f"Unsupported file type: {uploaded_file.name}. Please upload a PDF, TXT, DOCX, or MD file.")
    
    if len(extracted_texts) > 0:
        return extracted_texts
    elif len(uploaded_files) > 0:
        st.error(f"Unknown error!! Couldn't extract any text from uploaded documents.")


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=250,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks_with_ids: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"/n /n ####################### Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("No new documents to add")

    return db


def calculate_chunk_ids(chunks):

    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        current_page_id = f"{source}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def main():
    # load_dotenv()
    # print(f"Just to test .env variable, \nhere is the content of my DUMMY_KEY: {os.getenv('DUMMY_KEY')}")

    st.set_page_config(
        page_title="ResRAG App.",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.header("ResRAG App!")

    clear_database()

    os.makedirs(CHROMA_PATH)

    uploaded_files = list()
    uploaded_files = st.file_uploader("Please upload your resume files here!", accept_multiple_files=True)

    if len(uploaded_files) > 0:
        docs = oldstyle_reader(uploaded_files)
        # print(f"/n/n A sample doc: {docs[0]}")
        chunks = split_documents(docs)
        # Calculate Page IDs.
        chunks_with_ids = calculate_chunk_ids(chunks)
        print(f"Text split into {len(chunks_with_ids)} chunks")
        print(f"/n/n A sample chunck: {chunks_with_ids[13]}")

        vector_db = add_to_chroma(chunks_with_ids) 

        query_text = st.text_input(
            "Enter your question from the resumes database: "
        )

        if query_text:

            # Search the DB.
            results = vector_db.similarity_search_with_score(query_text, k=5)

            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
            # print(prompt)

            model = Ollama(model="llama3.2")
            response_text = model.invoke(prompt)

            sources = [doc.metadata.get("id", None) for doc, _score in results]
            formatted_response = f"Response: {response_text}\n Sources: {sources}"
            print(formatted_response)
            st.write("Here is the results: ", formatted_response)

            


if __name__ == "__main__":
    main()