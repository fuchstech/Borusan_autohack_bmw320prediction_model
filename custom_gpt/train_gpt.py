from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import os
#source https://www.datascienceengineer.com/blog/post-multiple-pdfs-with-gpt

# OpenAI platform key
os.environ["OPENAI_API_KEY"] = "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"
# Load pdf file and split into chunks
loader = PyPDFLoader(r"C:\Users\dest4\Desktop\autohackmatiricie\BMW_DATA\E46\1 models.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pages = loader.load_and_split(text_splitter)
# Prepare vector store
directory = 'index_store'
vector_index = Chroma.from_documents(pages, OpenAIEmbeddings(), persist_directory=directory)
vector_index.persist() # actually the Chroma client automatically persists the indexes when it is disposed - however better save then sorry :-)
# Prepare the retriever chain
retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k":6})
qa_interface = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
# First query
print(qa_interface("Where is the location of fuses on BMW E46"))
#Adding additional docs
