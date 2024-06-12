from fastapi import FastAPI, Query
from  dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.llms import OpenAI
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama



import shutil

load_dotenv()
# os.environ["OPEN_API_KEY"]=os.getenv("OPEN_API_KEY")


from langchain.document_loaders import PyPDFLoader


app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"
)


db_used="faiss" # "chroma"
llm_used="open-ai" # phi3,open-ai
model_name="gpt-3.5-turbo"

@app.post("/upload/")
def upload_and_embed(file_name: str = Query(..., title="File Path")):

    if os.path.exists('./faiss_db'):
        shutil.rmtree('./faiss_db')
        print("folder deleted")
    if os.path.exists('./chroma_db'):
        chroma_db = Chroma(persist_directory="./chroma_db/")

        # Delete documents one by one
        for document in chroma_db.get()['ids']:
            chroma_db._collection.delete(ids=document)
            print(f"Deleted document: {document}") 

    print(file_name)
    loader=PyPDFLoader(file_name)
    text_documents=loader.load()

    text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents=text_splitter.split_documents(text_documents)
    if db_used=="faiss":
        db=FAISS.from_documents(documents,OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY")))
        db.save_local("./faiss_db")

    else:
        db = Chroma.from_documents(documents,OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY")),persist_directory="./chroma_db")


    return "upload is successfull"

@app.post("/search/")
def search(input_text: str = Query(..., title="Input Text")):

    print(input_text)
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. Keep the output response in less than 100 words .
    <context>
    {context}
    </context>
    Question: {input}""")

    try:
        if llm_used=="open-ai":
            if model_name=="gpt-4o":
                llm=ChatOpenAI(openai_api_key=os.environ["OPEN_API_KEY"],temperature=0.6,model="gpt-4o")
            else:
                llm=OpenAI(openai_api_key=os.environ["OPEN_API_KEY"],temperature=0.1)
        elif llm_used=="phi3":
            llm=Ollama(model="phi3")
    except Exception as e:
        print(f"error while loading model: {e} ")
    if db_used=="faiss":
        db_search= FAISS.load_local("./faiss_db", OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY")),allow_dangerous_deserialization=True)
    elif db_used=="chroma":
        db_search= Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY")))

    retriever=db_search.as_retriever()
    document_chain=create_stuff_documents_chain(llm,prompt)
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    print("test5")
    try:
        response=retrieval_chain.invoke({"input":f"{input_text}"})
    except Exception as e:
        print(f"Unexpected error: {e}") 
    print("test6")

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8800)