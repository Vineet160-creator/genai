import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import uuid
import os
import pandas as pd
from docx import Document
import requests
from requests.auth import HTTPBasicAuth
import ollama  # For local LLM

# --- Step 1: Confluence Attachment Downloader ---
def download_confluence_attachments(base_url, page_id, username, password_or_token, download_dir):
    url = f"{base_url}/rest/api/content/{page_id}/child/attachment"
    auth = HTTPBasicAuth(username, password_or_token)

    response = requests.get(url, auth=auth)
    response.raise_for_status()

    attachments = response.json().get('results', [])
    downloaded_files = []

    for attach in attachments:
        file_url = attach['_links']['download']
        full_url = f"{base_url.split('/wiki')[0]}/wiki{file_url}"
        filename = attach['title']
        file_path = os.path.join(download_dir, filename)

        with requests.get(full_url, auth=auth, stream=True) as f:
            with open(file_path, 'wb') as out_file:
                out_file.write(f.content)
            downloaded_files.append(file_path)

        print(f"Downloaded from Confluence: {filename}")

    return downloaded_files

# --- Step 2a: Extract from PDF ---
def extract_chunks_from_pdf(filepath, chunk_size=300):
    chunks, metadatas = [], []
    doc_name = os.path.basename(filepath)
    with fitz.open(filepath) as doc:
        for page_num, page in enumerate(doc, start=1):
            words = page.get_text().split()
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                chunks.append(chunk)
                metadatas.append({"type": "pdf", "page": page_num, "source": doc_name, "path": filepath})
    return chunks, metadatas

# --- Step 2b: Extract from XLSX ---
def extract_chunks_from_xlsx(filepath, chunk_size=300):
    chunks, metadatas = [], []
    doc_name = os.path.basename(filepath)
    xl = pd.ExcelFile(filepath)
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name).fillna('')
        for row_idx, row in df.iterrows():
            words = ' '.join(str(cell) for cell in row if cell).split()
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                chunks.append(chunk)
                metadatas.append({"type": "xlsx", "sheet": sheet_name, "row": row_idx + 1, "source": doc_name, "path": filepath})
    return chunks, metadatas

# --- Step 2c: Extract from DOCX ---
def extract_chunks_from_docx(filepath, chunk_size=300):
    chunks, metadatas = [], []
    doc_name = os.path.basename(filepath)
    doc = Document(filepath)
    all_text = ' '.join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
    words = all_text.split()
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        metadatas.append({"type": "docx", "source": doc_name, "path": filepath})
    return chunks, metadatas

# --- Step 3: Embedding model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Step 4: ChromaDB setup ---
from chromadb.config import Settings
client = chromadb.PersistentClient(path="D:/GenAIProjects/ChromaDB/chroma_storage")
collection = client.get_or_create_collection("pdf_docs")

# --- Step 5: Download from Confluence ---
confluence_config = {
    "base_url": "https://vineetmishra160.atlassian.net/wiki",
    "page_id": "131242",
    "username": "vineetmishra160@gmail.com",
    "password_or_token": "ATATT3xFfGF0c15wVg0uGkGBkMmvMf2FJ236Hn9YYdeBGSapiCC3AjkZFHHlLB34N2rZ5wXikM0saNCpxxJ5I_7lCJ6PSW9miWEsuVY2uc5_jVuZAa3zKBuv6V2Ec5RB_aYh2GUW0nKST_g3ZGVyAg7mwdFqtBxh0Y73aFI4oc-0ks4HHLMsu4s=7A8A6554"  # REPLACE THIS
}
data_folder = r"D:\GenAIProjects\ChromaDB\confluencedowloads"

try:
    os.makedirs(data_folder, exist_ok=True)
    download_confluence_attachments(
        confluence_config["base_url"],
        confluence_config["page_id"],
        confluence_config["username"],
        confluence_config["password_or_token"],
        data_folder
    )
except Exception as e:
    print(f"Confluence download failed: {e}")

# --- Step 6: Index files ---
data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith((".pdf", ".xlsx", ".docx"))]
existing_metadatas = collection.get(include=["metadatas"])["metadatas"]
indexed_files = {meta["source"] for meta in existing_metadatas if "source" in meta}

for filepath in data_files:
    doc_name = os.path.basename(filepath)
    if doc_name in indexed_files:
        print(f"Skipping already indexed file: {doc_name}")
        continue

    print(f"Processing: {filepath}")
    try:
        if filepath.endswith(".pdf"):
            chunks, metadatas = extract_chunks_from_pdf(filepath)
        elif filepath.endswith(".xlsx"):
            chunks, metadatas = extract_chunks_from_xlsx(filepath)
        elif filepath.endswith(".docx"):
            chunks, metadatas = extract_chunks_from_docx(filepath)
        else:
            continue

        embeddings = model.encode(chunks).tolist()
        ids = [f"chunk_{uuid.uuid4().hex}" for _ in chunks]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    except Exception as e:
        print(f"Failed to process {filepath}: {e}")

print("✅ All new files indexed successfully.")

# --- Step 7: Query with RAG using Ollama ---
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    filter_filename = input("Enter a specific filename to filter by (or press Enter for all): ").strip()
    query_embedding = model.encode([query]).tolist()

    if filter_filename:
        filter_dict = {"source": {"$eq": filter_filename}}
    else:
        filter_dict = None

    results = collection.query(query_embeddings=query_embedding, n_results=5, where=filter_dict)
     # --- Display results ---
    print("\nTop Matching Passages:")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        source = meta.get("source")
        doc_type = meta.get("type", "unknown")
        context_info = f"File: {source} | Type: {doc_type}"

        if doc_type == "pdf":
            context_info += f" | Page: {meta.get('page')}"
        elif doc_type == "xlsx":
            context_info += f" | Sheet: {meta.get('sheet')} | Row: {meta.get('row')}"
        elif doc_type == "docx":
            context_info += " | Paragraph chunk"

        print(f"\n[{i+1}] {context_info}:\n{doc}")
    contexts = results["documents"][0]

    context_text = "\n\n".join(contexts)
    prompt = f"""Answer the following question using the provided context. If the context is insufficient, say so.

Context:
{context_text}

Question:
{query}
"""

    response = ollama.chat(
        model='phi:latest',
        messages=[{"role": "user", "content": prompt}]
    )

    print("\n🧠 Answer:")
    print(response["message"]["content"])
