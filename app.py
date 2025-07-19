
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

PROMPT_TEMPLATE = """
You are an HR assistant analyzing resumes against a job search query.

SEARCH QUERY: {question}

RESUME CONTEXT: 
{context}

INSTRUCTIONS:
1. Analyze EACH resume separately - do not mix information between resumes
2. For each resume, extract ONLY the information that directly relates to the search query
3. Score each resume independently based on how well it matches the query requirements
4. Be specific about which skills/experiences you found in each individual resume

RESPONSE FORMAT:
For each resume found in the context, respond with:

**Resume Match #[number]**
**Source:** [filename from context]
**Match Score:** [1-10]/10
**Key Strengths:** [2-3 specific qualifications found in THIS resume only]
**Match Explanation:** [Why THIS specific resume matches/doesn't match the query]
**Experience Level:** [Junior/Mid/Senior based on THIS resume's content]
**Location:** [If mentioned in THIS resume, look for their most recent location]

SCORING GUIDELINES:
- 9-10: Excellent match - meets most/all requirements
- 7-8: Good match - meets key requirements with minor gaps  
- 5-6: Moderate match - some relevant skills but missing important ones
- 3-4: Weak match - few relevant qualifications
- 1-2: Poor match - minimal or no relevant qualifications

CRITICAL RULES:
- Use ONLY information actually present in each individual resume
- Do NOT combine or mix information from different resumes
- If information is not clearly stated in a resume, do not assume it exists
- Keep explanations concise and factual
- Focus only on skills/experiences that relate to the search query

Begin your analysis:
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
        chunk_size=2500,
        chunk_overlap=1000,
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
        initial_sidebar_state="collapsed",
    )

    st.header("ResRAG App!")

    clear_database()
    # os.makedirs(CHROMA_PATH)

    # Ensure the DB directory exists
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH, exist_ok=True)

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

            # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

            context_text = ""
            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get("source", f"Resume_{i}")
                context_text += f"RESUME {i} (Source: {source}):\n{doc.page_content}\n\n---\n\n"

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