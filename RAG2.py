# from fastapi import FastAPI,Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# #from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama import OllamaEmbeddings
# from langchain_ollama import ChatOllama
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate




# class RequestQuery(BaseModel):
#     question:str

# app=FastAPI()
# templates = Jinja2Templates(directory="templates")




# #Load the data
# loader = PyPDFLoader(r"C:\Users\DELL\Documents\Python_Classes\LLM_application\Nandyal-Travel-Guide-by-ixigo.pdf")
# pages=loader.load()


# #split the data
# textsplitter =RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
# docs=textsplitter.split_documents(pages)

# #embedding model
# embedding=OllamaEmbeddings(model="nomic-embed-text")

# #loading the data to vector store
# vectorstores=Chroma.from_documents(documents=docs,embedding=embedding,persist_directory="./Chroma")

# #retriever
# retriever=vectorstores.as_retriever()

# #llm model which is ollama 3.2 latest
# llm = ChatOllama(model="llama3.2:latest")

# #prompt template
# custom_prompt=PromptTemplate(
#     template="""As a data analyzer
#     Context:{context}
#     Question:{question}
#     Answer:
#     """,
#     input_variables=["context","question"]
# )

# rag_chain=RetrievalQA.from_chain_type(
#             llm=llm,retriever=retriever,
#              return_source_documents=True,
#              chain_type_kwargs={"prompt":custom_prompt})

# # @app.post("/query")
# # def get_answer(request:RequestQuery):
# #     result=rag_chain(request.question)
# #     return {
# #         "question":request.question,
# #         "answer":result['result'],
# #         "sources": [doc.metadata for doc in result["source_documents"]]
# #     }


# @app.post("/query")
# def get_answer(request: RequestQuery):
#     try:    
#         result = rag_chain.invoke({"query": request.question})
#         #add the 
#         return {
#             "question": request.question,
#             "answer": result["result"],
#             "sources": [doc.metadata for doc in result["source_documents"]]
#         }
#     except Exception as e:
#         return {"error": str(e)}

# # @app.get("/")
# # def root():
# #     return {"message": "RAG API is running!"}

# # === HTML frontend ===
# @app.get("/", response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})






from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from pathlib import Path
import shutil

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI()

# -----------------------------
# Templates (HTML frontend)
# -----------------------------
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Global Variables
# -----------------------------
UPLOAD_DIR = Path("./uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

# Embedding and LLM models
embedding = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:latest")

# Prompt template
custom_prompt = PromptTemplate(
    template="""As a data analyzer
    Context:{context}
    Question:{question}
    Answer:
    """,
    input_variables=["context", "question"]
)

# Original PDF path
ORIGINAL_PDF = Path(r"C:\Users\DELL\Documents\Python_Classes\LLM_application\Nandyal-Travel-Guide-by-ixigo.pdf")

# Global vectorstore
vectorstores = None

# -----------------------------
# Load original PDF into vectorstore
# -----------------------------
def load_vectorstore(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = textsplitter.split_documents(pages)
    store = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory="./Chroma")
    return store

# Load original PDF at startup
vectorstores = load_vectorstore(ORIGINAL_PDF)
retriever = vectorstores.as_retriever()
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# -----------------------------
# Pydantic model for query
# -----------------------------
class RequestQuery(BaseModel):
    question: str

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    global vectorstores, retriever, rag_chain
    
    # Save uploaded file
    file_location = UPLOAD_DIR / file.filename
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load uploaded PDF into vectorstore
    vectorstores = load_vectorstore(file_location)
    retriever = vectorstores.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    return {"filename": file.filename, "message": "File uploaded and processed successfully!"}


@app.post("/query")
def get_answer(request: RequestQuery):
    try:
        result = rag_chain.invoke({"query": request.question})
        return {
            "question": request.question,
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }
    except Exception as e:
        return {"error": str(e)}
