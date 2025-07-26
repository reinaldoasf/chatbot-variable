from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in aswereing question about a pizza restaurant

Here are some relevante reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n-------------------------------------------------")
    question = input("Ask a question (or press q to quit):")
    print("\n\n")
    if question == 'q':
        break

    result = chain.invoke({"reviews": [], "question":"What is the best pizza place in town?"})
    print(result)