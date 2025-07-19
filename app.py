from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
import docx2txt2
from bs4 import BeautifulSoup
from markdown import markdown
import re
import shutil
import base64
from langchain_core.documents import Document
from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate



CHROMA_PATH = "./DBPATH"
DATA_PATH = "./DATA"

# PROMPT_TEMPLATE = """
# You are an HR assistant analyzing resumes against a job search query.

# SEARCH QUERY: {question}

# RESUME CONTEXT: 
# {context}

# INSTRUCTIONS:
# 1. Analyze EACH resume separately - do not mix information between resumes
# 2. For each resume, extract ONLY the information that directly relates to the search query
# 3. Score each resume independently based on how well it matches the query requirements
# 4. Be specific about which skills/experiences you found in each individual resume
# 5. Identify and highlight any mismatches or gaps in requirements

# RESPONSE FORMAT:
# For each resume found in the context, respond with:

# **Resume Match #[number]**
# **Source:** [filename from context]
# **Match Score:** [1-10]/10
# **Key Strengths:** [2-3 specific qualifications found in THIS resume only]
# **Match Explanation:** [Why THIS specific resume matches/doesn't match the query]
# **Experience Level:** [Junior/Mid/Senior based on THIS resume's content]
# **Location:** [If mentioned in THIS resume]
# **Potential Mismatches:** [List any requirements from the query that this resume does NOT meet, such as location, degree, specific skills, experience level, etc.]

# SCORING GUIDELINES:
# - 9-10: Excellent match - meets most/all requirements
# - 7-8: Good match - meets key requirements with minor gaps  
# - 5-6: Moderate match - some relevant skills but missing important ones
# - 3-4: Weak match - few relevant qualifications
# - 1-2: Poor match - minimal or no relevant qualifications

# CRITICAL RULES:
# - Use ONLY information actually present in each individual resume
# - Do NOT combine or mix information from different resumes
# - If information is not clearly stated in a resume, do not assume it exists
# - Keep explanations concise and factual
# - Focus only on skills/experiences that relate to the search query
# - Always identify mismatches even if the overall score is high

# After analyzing all individual resumes, provide:

# **SUMMARY ANALYSIS**
# **Top Recommended Candidate:** [Best matching resume filename]
# **Recommendation Score:** [Score of the best candidate]/10
# **Why This is the Best Match:** [2-3 key reasons why this resume stands out above others]
# **Main Strengths:** [Top 3 strengths of the recommended candidate]
# **Areas of Concern:** [Any significant mismatches or gaps in the top candidate, if any]

# Begin your analysis:
# """


PROMPT_TEMPLATE = """
You are an HR assistant analyzing resumes against a job search query. Before scoring, you must think carefully about what the role actually requires.

SEARCH QUERY: {question}

RESUME CONTEXT: 
{context}

ANALYSIS APPROACH:
1. **First, carefully analyze the search query** - What is the PRIMARY role being sought? What are the ESSENTIAL vs. NICE-TO-HAVE skills for this type of position?

2. **Think about role requirements** - Consider what someone in this role would actually do day-to-day. What skills are absolutely critical vs. what might be helpful but not necessary?

3. **Evaluate each candidate's PRIMARY expertise** - What is this person's main professional identity? How well does their core expertise align with what's actually needed?

4. **Apply logical reasoning** - Don't assume that "more skills = better candidate." A specialist who excels in the core area may be better than a generalist with surface-level knowledge across many areas.

INSTRUCTIONS:
1. Start with a brief analysis of what the search query is really asking for
2. For each resume, identify the candidate's primary professional expertise first
3. Then assess how well that expertise matches the actual role requirements
4. Score based on role fit and core skill alignment, not just skill quantity
5. Use logical reasoning - consider what the hiring manager actually needs

RESPONSE FORMAT:

**QUERY ANALYSIS:**
[1-2 sentences analyzing what role is being sought and what skills are truly essential for success in this position]

For each resume found in the context:

**Resume Match #[number]**
**Source:** [filename from context]
**Candidate's Primary Expertise:** [What is this person's main professional identity/specialization?]
**Role Alignment Assessment:** [How well does their core expertise match what's actually needed?]
**Match Score:** [1-10]/10
**Essential Skills Present:** [Skills that directly relate to the core job requirements]
**Match Reasoning:** [Logical explanation of why this score makes sense for this specific role]
**Experience Level:** [Junior/Mid/Senior based on content]
**Location:** [If mentioned]
**Additional Considerations:** [Any other relevant factors - positive or negative]

SCORING PHILOSOPHY:
- A candidate whose core expertise perfectly matches the role should score higher than someone with many tangential skills
- Consider what success in this role actually requires
- Don't penalize candidates for lacking skills that aren't central to the role's primary responsibilities
- Value depth in relevant areas over breadth in unrelated ones

After analyzing all resumes:

**SUMMARY ANALYSIS**
**Role Focus:** [What the search query is fundamentally asking for]
**Top Recommended Candidate:** [Best matching resume filename]
**Recommendation Score:** [Score]/10
**Selection Reasoning:** [Why this candidate's core expertise best fits what the role actually requires]
**Key Advantages:** [What makes this candidate the logical choice for this specific role]
**Fit Assessment:** [How well their primary expertise aligns with the role's true needs]

Think logically about role requirements and candidate fit. Begin your analysis:
"""



def create_download_link(file_path, file_name):
    """Create a download link for a file"""
    try:
        with open(file_path, "rb") as f:
            bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        
        # Determine file type for appropriate download
        file_ext = file_name.split('.')[-1].lower()
        if file_ext == 'pdf':
            mime_type = 'application/pdf'
        elif file_ext in ['doc', 'docx']:
            mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        elif file_ext == 'txt':
            mime_type = 'text/plain'
        elif file_ext == 'md':
            mime_type = 'text/markdown'
        else:
            mime_type = 'application/octet-stream'
            
        href = f'<a href="data:{mime_type};base64,{b64}" download="{file_name}" target="_blank">📄 {file_name}</a>'
        return href
    except Exception as e:
        return f"❌ File not found: {file_name}"


def save_uploaded_file(uploaded_file):
    """Save uploaded file to DATA directory and return the file path"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=True)
    
    file_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


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
        # Save the uploaded file to DATA directory
        file_path = save_uploaded_file(uploaded_file)
        print(f"Saved file: {uploaded_file.name} to {file_path}")

        text = ""
        if uploaded_file.name.endswith('.pdf'):
            uploaded_file.seek(0)  # Reset file pointer
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
            text = clean_text_from_raw_pdf(text)
            extracted_texts.append(Document(page_content=text, metadata={
                "source": str(uploaded_file.name),
                "file_path": file_path
            }))

        elif uploaded_file.name.endswith('.txt'):
            uploaded_file.seek(0)  # Reset file pointer
            text = uploaded_file.read().decode("utf-8")
            extracted_texts.append(Document(page_content=text, metadata={
                "source": str(uploaded_file.name),
                "file_path": file_path
            }))

        elif uploaded_file.name.endswith('.docx'):
            uploaded_file.seek(0)  # Reset file pointer
            text = docx2txt2.extract_text(uploaded_file)
            extracted_texts.append(Document(page_content=text, metadata={
                "source": str(uploaded_file.name),
                "file_path": file_path
            }))
            
        elif uploaded_file.name.endswith('.md'):
            uploaded_file.seek(0)  # Reset file pointer
            text = uploaded_file.read().decode("utf-8")
            text = markdown_to_text(text)
            extracted_texts.append(Document(page_content=text, metadata={
                "source": str(uploaded_file.name),
                "file_path": file_path
            }))

        else:
            st.error(f"Unsupported file type: {uploaded_file.name}. Please upload a PDF, TXT, DOCX, or MD file.")
    
    if len(extracted_texts) > 0:
        print(f"Successfully extracted {len(extracted_texts)} documents")
        for doc in extracted_texts:
            print(f"Doc: {doc.metadata}")
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
    else:
        print("No new documents to add")

    return db


def calculate_chunk_ids(chunks):
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


def extract_best_match_from_response(response_text, results):
    """Extract the best matching resume filename from the AI response and create download link"""
    try:
        # Look for the top recommended candidate in the response
        lines = response_text.split('\n')
        best_match_filename = None
        
        for line in lines:
            if "Top Recommended Candidate:" in line or "**Top Recommended Candidate:**" in line:
                # Extract filename from the line
                parts = line.split(":")
                if len(parts) > 1:
                    best_match_filename = parts[1].strip().replace("*", "").strip()
                    print(f"Found best match filename: '{best_match_filename}'")
                break
        
        # If we found a best match filename, find the corresponding file path
        if best_match_filename:
            # Get unique documents from results to avoid duplicates
            unique_docs = {}
            for doc, score in results:
                source = doc.metadata.get("source")
                if source not in unique_docs:
                    unique_docs[source] = doc
            
            print(f"Available sources: {list(unique_docs.keys())}")
            
            # Try to find exact match or partial match
            for source, doc in unique_docs.items():
                if source == best_match_filename or best_match_filename in source:
                    file_path = doc.metadata.get("file_path")
                    print(f"Found matching source: {source}, file_path: {file_path}")
                    if file_path and os.path.exists(file_path):
                        return create_download_link(file_path, source)
                    else:
                        # Try to find in DATA folder as backup
                        backup_path = os.path.join(DATA_PATH, source)
                        if os.path.exists(backup_path):
                            return create_download_link(backup_path, source)
        
        return None
    except Exception as e:
        print(f"Error extracting best match: {e}")
        return None


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def main():
    st.set_page_config(
        page_title="ResRAG App.",
        initial_sidebar_state="collapsed",
    )

    st.header("ResRAG App!")

    # Ensure the DB and DATA directories exist
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=True)

    # Add option to clear data
    with st.sidebar:
        if st.button("Clear Database & Files"):
            clear_database()
            if os.path.exists(DATA_PATH):
                shutil.rmtree(DATA_PATH)
            os.makedirs(CHROMA_PATH, exist_ok=True)
            os.makedirs(DATA_PATH, exist_ok=True)
            st.success("Database and files cleared!")

    uploaded_files = list()
    uploaded_files = st.file_uploader("Please upload your resume files here!", accept_multiple_files=True)

    if len(uploaded_files) > 0:
        docs = oldstyle_reader(uploaded_files)
        chunks = split_documents(docs)
        chunks_with_ids = calculate_chunk_ids(chunks)
        print(f"Text split into {len(chunks_with_ids)} chunks")
        print(f"/n/n A sample chunk: {chunks_with_ids[0] if chunks_with_ids else 'No chunks'}")

        vector_db = add_to_chroma(chunks_with_ids) 

        query_text = st.text_input(
            "Enter your question from the resumes database: "
        )

        if query_text:
            # Search the DB.
            results = vector_db.similarity_search_with_score(query_text, k=5)

            context_text = ""
            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get("source", f"Resume_{i}")
                context_text += f"RESUME {i} (Source: {source}):\n{doc.page_content}\n\n---\n\n"

            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)

            model = Ollama(model="llama3.2")
            response_text = model.invoke(prompt)

            # Extract best match and create download link
            best_match_link = extract_best_match_from_response(response_text, results)

            sources = [doc.metadata.get("id", None) for doc, _score in results]
            
            st.write("### Analysis Results")
            st.write(response_text)
            
            # Display the download link for the best match
            if best_match_link:
                st.write("### 📋 Best Match Resume:")
                st.markdown(best_match_link, unsafe_allow_html=True)
            else:
                st.warning("Could not create download link for best match. Check console for debugging info.")
            
            # Show all available resume files with download links
            st.write("### 📁 All Resume Files:")
            
            # Get unique documents to avoid duplicates
            unique_docs = {}
            for doc, score in results:
                source = doc.metadata.get("source")
                if source and source not in unique_docs:
                    unique_docs[source] = doc
            
            if unique_docs:
                for source, doc in unique_docs.items():
                    file_path = doc.metadata.get("file_path")
                    if file_path and os.path.exists(file_path):
                        link = create_download_link(file_path, source)
                        st.markdown(f"• {link}", unsafe_allow_html=True)
                    else:
                        # Try backup path
                        backup_path = os.path.join(DATA_PATH, source)
                        if os.path.exists(backup_path):
                            link = create_download_link(backup_path, source)
                            st.markdown(f"• {link}", unsafe_allow_html=True)
                        else:
                            st.write(f"• ❌ {source} (file not found)")
            else:
                st.warning("No resume files found in results.")
            
            st.write(f"**Sources:** {sources}")


if __name__ == "__main__":
    main()