# from langchain_community.document_loaders import TextLoader
# from langchain_core.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from observability.langfuse_setup import langfuse
import time

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




# question = "How many annual leave days do employees get?"

# # trace = langfuse.trace(
# #     name="rag-question"
# # )

# # trace.event(
# #     name="user-question",
# #     input={
# #         "question": question
# #     }
# # )



# trace = langfuse.trace(
#     name="rag-eval-project",
#     input={
#         "question": question
#     }
# )





# docs = retriever.invoke(question)

# context = "\n".join(
#     [doc.page_content for doc in docs]
# )

# # trace.event(
# #     name="retrieved-context",
# #     output={
# #         "context": context
# #     }
# # )


# trace.update(
#     metadata={
#         "retrieved_context": context
#     }
# )




# print("\nRetrieved Context:")
# print(context)

# # prompt = f"""
# # Answer ONLY using the context.

# # Context:
# # {context}

# # Question:
# # {question}

# # If answer is not found say:
# # I don't know.
# # """


# prompt = f"""
# You are an HR assistant.

# Use the context below to answer the question.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """



# print("\nPrompt:")
# print(prompt)


# response = llm.invoke(prompt)



# # trace.event(
# #     name="qa-test",
# #     input={
# #         "question": question
# #     },
# #     output={
# #         "answer": response.content
# #     },
# #     metadata={
# #         "expected_answer": "25 days",
# #         "test_type": "retrieval_accuracy"
# #     }
# # )


# trace.update(
#     output={
#         "answer": response.content
#     }
# )






# # trace.event(
# #     name="llm-answer",
# #     output={
# #         "answer": response.content
# #     }
# # )


# trace.update(
#     output={
#         "answer": response.content
#     },
#     metadata={
#         "expected_answer": "25 days",
#         "test_type": "retrieval_accuracy",
#         "model": "llama3.2"
#     }
# )






# print("\nQuestion:")
# print(question)

# print("\nAnswer:")
# print(response.content)
# langfuse.flush()



question = "How many annual leave days do employees get?"

# with langfuse.start_as_current_observation(
#     name="rag-eval-project"
# ):

#     docs = retriever.invoke(question)

#     context = "\n".join(
#         [doc.page_content for doc in docs]
#     )

#     print("\nRetrieved Context:")
#     print(context)

#     prompt = f"""
# You are an HR assistant.

# Use the context below to answer the question.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     print("\nPrompt:")
#     print(prompt)

#     response = llm.invoke(prompt)

#     # Log Question
#     langfuse.create_event(
#         name="question",
#         body={
#             "question": question
#         }
#     )

#     # Log Retrieved Context
#     langfuse.create_event(
#         name="retrieved_context",
#         body={
#             "context": context
#         }
#     )

#     # Log Answer
#     langfuse.create_event(
#         name="answer",
#         body={
#             "answer": response.content,
#             "expected_answer": "25 days",
#             "test_type": "retrieval_accuracy",
#             "model": "llama3.2"
#         }
#     )

#     print("\nQuestion:")
#     print(question)

#     print("\nAnswer:")
#     print(response.content)

# langfuse.flush()



# with langfuse.start_as_current_observation(
#     name="rag-eval-project"
# ):

#     docs = retriever.invoke(question)

#     context = "\n".join(
#         [doc.page_content for doc in docs]
#     )

#     start_time = time.time()

#     response = llm.invoke(prompt)

#     latency = round(time.time() - start_time, 2)

#     langfuse.create_event(
#         name="question",
#         input={
#             "question": question
#         }
#     )

#     langfuse.create_event(
#         name="retrieved_context",
#         output={
#             "context": context
#         }
#     )

#     langfuse.create_event(
#         name="answer",
#         output={
#             "answer": response.content
#         },
#         metadata={
#             "expected_answer": "25 days",
#             "test_type": "retrieval_accuracy",
#             "model": "llama3.2",
#             "latency_seconds": latency
#         }
#     )

# langfuse.flush()




question = "How many annual leave days do employees get?"

with langfuse.start_as_current_observation(
    name="rag-eval-project"
):

    # Retrieve documents
    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    print("\nRetrieved Context:")
    print(context)

    # Build prompt
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

    # Measure latency
    start_time = time.time()

    response = llm.invoke(prompt)

    latency = round(time.time() - start_time, 2)

    # Log question
    langfuse.create_event(
        name="question",
        input={
            "question": question
        }
    )

    # Log retrieved context
    langfuse.create_event(
        name="retrieved_context",
        output={
            "context": context
        }
    )

    # Log answer
    langfuse.create_event(
        name="answer",
        output={
            "answer": response.content
        },
        metadata={
            "expected_answer": "25 days",
            "test_type": "retrieval_accuracy",
            "model": "llama3.2",
            "latency_seconds": latency
        }
    )

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(response.content)

langfuse.flush()


print("Trace ID:", langfuse.get_current_trace_id())