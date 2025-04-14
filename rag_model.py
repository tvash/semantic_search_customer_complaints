#Takes some time to run 
from langchain.chat_models import init_chat_model
#USE TO GET THE TEXT SPLITTING 
from langchain_text_splitters import RecursiveCharacterTextSplitter
#GRAB DOCUMENTS WHICH NEEDS TO BE CHUNDKED AND EMBEDDING
from langchain_core.documents import  Document
from typing_extensions import List, Dict
from weaviate 
from langchain.vectorstores import Weaviate
import getpass
import os 


import FAISS #PLACEHOLDER OPTION NEEDS TO BE REPLACED



"""
This code will accept documents and create type extensions which will later be loaded into the a vector database
FAISS is a simple good to go option to store it but compare the different types of VDB
"""

complaints = ["something1", "something2"]


print("SUCCESSFULY")
