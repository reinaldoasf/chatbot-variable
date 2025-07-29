import ollama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

def load_pdf_documents(pdf_path):
    try:
        if os.path.exists(pdf_path):
            loader = PyPDFLoader(pdf_path)
        return loader.load()
    except Exception as e:
        raise (e)

def split_documents(pages, chunk_size):
    """Split document pages in chunks of size = chunk_size"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200
        )
        return text_splitter.split_documents(pages)
    except Exception as e:
        raise (e)

def create_vector_store(split_docs):
    """Create vector store using Sentence transformers embeddings
        stores it using faiss library to store vectors"""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    document_texts = [doc.page_content for doc in split_docs]
    embeddings = embedder.encode(document_texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    
    return index, document_texts, embedder

def retrieve_context(query, embedder, index, documents, k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return [documents[i] for i in indices[0]]

def generate_answer_with_ollama(querry, context, model_name):
    formated_context = "\n".join(context)
    prompt = f"""
    You are an expert document retriever information. Use this context to answer the question:
        Context: {formated_context}
        
        Question: {querry}

    Answer in detail using only the provided context
    """
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            'temperature': 0.3,
            'max_tokens': 2000
        }
    )
    return (response['response'])

def main(pdf_path, model_name):
    pages = load_pdf_documents(pdf_path)
    split_docs = split_documents(pages, chunk_size=1000)

    indexes, document_texts, embedder = create_vector_store(split_docs)

    querry = input(f"\n\nDigite o que gostaria de saber sobre o documento(ou q para sair)\n{pdf_path}\n\n")
    while querry != 'q':
        context = retrieve_context(querry, embedder, indexes, document_texts)
        answer = generate_answer_with_ollama(querry, context, model_name)
        print('----------------------------------')
        print(answer)
        print('----------------------------------')
        querry = input(f"\n\nDigite o que gostaria de saber sobre o documento(ou q para sair)\n{pdf_path}\n\n")
    print("Obrigado por interagir com nosso modelo.")

if __name__ == '__main__':
    pdf_path = "/home/reinaldoasf/Documents/DeepSeek-Artigo.pdf"
    model_name="llama3.2"
    main(pdf_path, model_name)



