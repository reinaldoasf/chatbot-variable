import os
import sys
import logging
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.readers.web import BeautifulSoupWebReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
load_dotenv()

USE_OLLAMA = True

DATA_DIR = './data/UBER'
PERSIST_DATA_DIR = './storage'

def configure_models(model="llama3.2", embeddings_model="mxbai-embed-large") -> None:
    """Configure LLM and embedding model by default using Ollama local environment.
        params:
            model (str): LLM model name.
            embeddings_model (str):  embeddings model name.
    """
    print("Loading models.")
    if USE_OLLAMA:
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding
        Settings.llm = Ollama(model=model, request_timeout=240)
        Settings.embed_model = OllamaEmbedding(model_name=embeddings_model)
    else:
        #TODO create a configuration to other paid APIs.
        pass
    print("Models setup complete.")

def load_documents():
    """Load documents from local storage or web."""
    print("Loading local_files")
    #TODO read files from list of paths from json
    local_documents = SimpleDirectoryReader(DATA_DIR).load_data()
    
    print(f"{len(local_documents)} local documents on path\n{DATA_DIR}")

    urls_to_load = [
        "https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial"
    ]#TODO replace for the json file reader.

    if urls_to_load:
        print(f"Loading {len(urls_to_load)} web pages.")
        loader_web = BeautifulSoupWebReader()
        web_docs = loader_web.load_data(urls=urls_to_load)
        print("Web documents are read")
    
    return local_documents + web_docs



def main(): #TODO add possibility of add arg to choose witch environment to build the storage.
    configure_models()
    
    if not os.path.exists(DATA_DIR):
        raise("Create and place data on ", DATA_DIR)
    
    documents = load_documents()

    if not os.path.exists(PERSIST_DATA_DIR):
        print("Creating new indexes")
        if not documents:
            raise("No document found as source of knowledge base")
        index = VectorStoreIndex.from_documents(documents=documents, show_progress=True)
        print(f"Saving index on {PERSIST_DATA_DIR}")
        index.storage_context.persist(persist_dir=PERSIST_DATA_DIR)
    else:
        print("Index found. Updating with new content")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DATA_DIR)
        index = load_index_from_storage(storage_context)
        refresh_info = index.refresh_ref_docs(documents, show_progress=True)
        print(f"Added: {sum(refresh_info.inserted_doc_ids)}")
        print(f"Added: {sum(refresh_info.updated_doc_ids)}")
        print(f"Added: {sum(refresh_info.updated_doc_ids)}")

        if any(refresh_info.values()):
            index.storage_context.persist(PERSIST_DATA_DIR)
            print("index updated")
        else:
            print("No new update on index storage")

if __name__ == '__main__':
    main()