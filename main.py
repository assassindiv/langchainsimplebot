import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
sentences = ["This is an example sentence", "Each sentence is converted"]
embeddings = model.encode(sentences)
print(embeddings)

print(" main.py is running")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
loader = TextLoader("knowledge.txt" )
documents = loader.load()
embeddings=HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
llm = ChatGroq(api_key=groq_api_key, model_name="mistral-saba-24b")
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever= retriever)


while True:
    query = input("Ask something (or type 'exit'): ")
    if query.lower() == 'exit':
        break
    print("Answer:", qa_chain.run(query))
