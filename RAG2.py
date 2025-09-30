from fastapi import FastAPI,Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate




class RequestQuery(BaseModel):
    question:str

app=FastAPI()
templates = Jinja2Templates(directory="templates")


#Load the data
loader = PyPDFLoader(r"C:\Users\DELL\Documents\Python_Classes\LLM_application\Nandyal-Travel-Guide-by-ixigo.pdf")
pages=loader.load()

#split the data
textsplitter =RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
docs=textsplitter.split_documents(pages)

#embedding model
embedding=OllamaEmbeddings(model="nomic-embed-text")

#loading the data to vector store
vectorstores=Chroma.from_documents(documents=docs,embedding=embedding,persist_directory="./Chroma")

#retriever
retriever=vectorstores.as_retriever()

#llm model which is ollama 3.2 latest
llm = ChatOllama(model="llama3.2:latest")

#prompt template
custom_prompt=PromptTemplate(
    template="""As a data analyzer
    Context:{context}
    Question:{question}
    Answer:
    """,
    input_variables=["context","question"]
)

rag_chain=RetrievalQA.from_chain_type(
            llm=llm,retriever=retriever,
             return_source_documents=True,
             chain_type_kwargs={"prompt":custom_prompt})

# @app.post("/query")
# def get_answer(request:RequestQuery):
#     result=rag_chain(request.question)
#     return {
#         "question":request.question,
#         "answer":result['result'],
#         "sources": [doc.metadata for doc in result["source_documents"]]
#     }


@app.post("/query")
def get_answer(request: RequestQuery):
    try:    
        result = rag_chain.invoke({"query": request.question})
        #add the 
        return {
            "question": request.question,
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }
    except Exception as e:
        return {"error": str(e)}

# @app.get("/")
# def root():
#     return {"message": "RAG API is running!"}

# === HTML frontend ===
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
