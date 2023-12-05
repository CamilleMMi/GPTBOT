import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

PERSIST = False

def load_data(file_path):
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([TextLoader(file_path)])
    else:
        index = VectorstoreIndexCreator().from_loaders([TextLoader(file_path)])
    return index

def create_file(file_name):
    with open(file_name, 'w') as file:
        print(f"Fichier {file_name} créé, fin de processus")
        sys.exit()

def main(query, file_name):
    full_file_path = os.path.join("data/", f"{file_name}.txt")

    if not os.path.exists(full_file_path):
        create_file(full_file_path)
    
    if PERSIST and os.path.exists("persist"):
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        index = load_data(full_file_path)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    while True:
        if not query:
            query = input("Prompt (q pour quitter): ")
        if query.lower() in ['quit', 'q', 'exit']:
            sys.exit()
        result = chain({"question": query, "chat_history": chat_history})
        print(result['answer'])

        chat_history.append((query, result['answer']))
        query = None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ./src/chatGptBot.py <file_name> <query>")
        sys.exit(1)

    file_name = sys.argv[1]
    query = sys.argv[2]
    main(query, file_name)
