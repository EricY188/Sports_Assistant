"""
Retrieval-Augmented Generation (RAG) module
"""

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

class SportsKnowledgeBase:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.initialize()
    
    def initialize(self):
        """Initialize or reload knowledge base"""
        loader = DirectoryLoader(
            '../data/knowledge_base',
            glob="**/*.*",
            loader_cls=PyPDFLoader,
            use_multithreading=True
        )
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        self.vector_store = FAISS.from_documents(
            documents=splits, 
            embedding=self.embeddings
        )
    
    def query(self, question: str, k: int = 3) -> str:
        """Retrieve relevant context from knowledge base"""
        docs = self.vector_store.similarity_search(question, k=k)
        return "\n\n".join([d.page_content for d in docs])
