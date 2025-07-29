import os
import sys
from generate_knowledgebase import configure_models

from llama_index.core import (
    StorageContext,
    load_index_from_storage
)

PERSIST_DIR = "./storage"

def main():
    if not PERSIST_DIR:
        raise(f"ERROR there is no knowledge base storage")
    
    configure_models(model="llama3:8b-instruct-q4_K_M")

    print("Loading Knowledge Base")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("Knowledge Base loaded successfully!")

    query_engine = index.as_query_engine(similarity_top_k=5, streaming=True)
    
    while True:
        query = input("Digite sua pergunta(Ou digite q para sair):")
        if query.lower() == 'q':
            break
        response = query_engine.query(query)
        print("Resposta: ", end="")
        response.print_response_stream()
        print("\n\n"+'-'*50+"\n\n")
    
    print("Obrigado por interagir com nosso chat.")

if __name__ == "__main__":
    main()

