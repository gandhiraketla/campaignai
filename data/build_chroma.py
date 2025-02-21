# build_chroma.py
import os
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.envutils import EnvUtils
DOCS_FOLDER = "c:\\demos\\CampaignAI"
PERSIST_DIR = "chroma_store"
os.environ["OPENAI_API_KEY"] = EnvUtils().get_required_env("OPENAI_API_KEY")

def build_chroma_index(docs_folder=DOCS_FOLDER, persist_dir=PERSIST_DIR):
    texts = []
    metadatas = []

    for fname in os.listdir(docs_folder):
        if fname.endswith(".txt"):
            path = os.path.join(docs_folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_text(raw_text)

            # Store each chunk as separate doc
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({"source": fname})

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Build local Chroma DB
    vectorstore = Chroma.from_texts(
        texts,
        embeddings,
        metadatas=metadatas,
        persist_directory=persist_dir
    )
    vectorstore.persist()

    print(f"Chroma index built with {len(texts)} chunks.")

if __name__ == "__main__":
    build_chroma_index()
