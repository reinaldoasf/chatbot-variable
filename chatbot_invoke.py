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

    #The follow line made a simple query response chat search information over knowledegebase
    #replace chat engine if it is the desired behavior
    #query_engine = index.as_query_engine(similarity_top_k=5, stream=True)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        similarity_top_k=5,
        system_prompt=(
            """Você é um agente de busca de informação
                Responda somente com a informação adquirida.
                """
        )
    )
    
    while True:
        query = input("Digite sua pergunta(Ou digite q para sair ou /reset para limpar o histórico da conversa):")
        if query.lower() == 'q':
            break
        if query.lower() == '/reset':
            chat_engine.reset()
            print("\n-------- Limpado historico de conversa -------\n")
            continue
        response_stream = chat_engine.stream_chat(query)
        print("Resposta do chat: ", end="")
        for token in response_stream.response_gen:
            print(token, end="")
        #response = query_engine.query(query)
        #print("Resposta: ", end="")
        #response.print_response_stream()
        print("\n\n"+'-'*50+"\n\n")
    
    print("Obrigado por interagir com nosso chat.")

if __name__ == "__main__":
    main()

