## ResRAG Project - RAG-Based Resume Search Tool

### Objective

Design and build a prototype application that allows an HR specialist to search and explore a pool of resumes using natural language queries. 
The app should use RAG (Retrieval-Augmented Generation) techniques to return relevant resume profiles, even if the query terms are not exact keyword matches.

---

### Requirements

#### 1. Core Functionality
- Upload CVs: Allow users to upload multiple CVs (PDF, DOCX, or TXT format).
- Data Extraction: Extract text content from the uploaded resumes (in PDF format).
- Indexing: Vectorize and store the resumes using embeddings (e.g., OpenAI).

<br>

> [!NOTE]
> - If the user uploads the resume twice, only one request should be sent to the LLM
for embedding, not two.
> - You can store metadata for each resume to provide better results when users
query them.
<br>

- Query Interface: 
    - Allow the HR user to enter natural language queries like:

    ```md
    > Looking for a data scientist with Python and TensorFlow experience in Germany.
    > Looking for a data scientist with Python and Pytorch experience in Iran.
    > Frontend developer with a React portfolio and a CS degree.
    ```

- RAG Retrieval:
    - Retrieve top-K relevant CVs using vector similarity.
    - Use an LLM to generate a short summary or justification of why each CV is a match.



#### 2. UI

- Build a basic Streamlit interface:
    - Upload resumes
    - Enter search query
    - Display matching CVs with summary/justification from the LLM


#### 3. Tech Stack
- Frontend: 
    - Python (Streamlit)

- Backend: 
    - Python (Streamlit)
- Database: 
    - Milvus / ChromaDB / Elasticsearch / or any vector DB | Also use Redis for caching.

- AI Capabilities: 
    - Use an embedding model (e.g., text-embedding-3-small) to vectorize CVs and queries, 
    - Use an LLM to generate human-readable summaries of match relevance

---
### Deliverables
- A GitHub repo (or ZIP) containing:
    - Dockerize Project
    - Requirements file (e.g., requirements.txt)
    - README with setup and usage instructions
    - A small sample dataset (3–5 anonymized CVs) along with corresponding queries, intended for demonstration purposes.



---

## Local Installation: (WSL2 Ubuntu 20.04)

1. First install `ollama`:

Download and extract the package:

```
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C /usr -xzf ollama-linux-amd64.tgz
```

Start Ollama:

```
ollama serve
```

In another terminal, verify that Ollama is running:

```
ollama -v
```

Then pull `llama3.2`

```
ollama pull llama3.2
```

2. Then clone the git reporistory:

```
git clone git@github.com:Amirhosein2c/ResRAG.git 
```

and enter the `ResRAG` directory

3. Now make a conda environment named `ResRag`:

```
conda create --name ResRag python=3.10
```

4. install the requirements using the following command:

```
pip install -r ./requirements.txt
```

5. Run the app using the command below:

```
streamlit run ./app.py
```

6. Upload dummy resume files from `./RESUME` directory.

7. Put your query in the dedicated box and check the results.

## Docker Run:

For running the app using `Docker` checkout to the `docker` branch of the git repository and follow the instructions in the related `README_DOCKER.md` file.