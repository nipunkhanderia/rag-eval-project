# from langchain_community.document_loaders import TextLoader
# from langchain_core.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader


# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.vectorstores import FAISS

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama import ChatOllama

# Load docs
loader = TextLoader("data/company_policy.txt")
docs = loader.load()

# Split docs
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

# Local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector store
db = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = db.as_retriever()

# Local LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)




question = "How many annual leave days do employees get?"

docs = retriever.invoke(question)

context = "\n".join(
    [doc.page_content for doc in docs]
)

print("\nRetrieved Context:")
print(context)

# prompt = f"""
# Answer ONLY using the context.

# Context:
# {context}

# Question:
# {question}

# If answer is not found say:
# I don't know.
# """


prompt = f"""
You are an HR assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""



print("\nPrompt:")
print(prompt)


response = llm.invoke(prompt)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(response.content)