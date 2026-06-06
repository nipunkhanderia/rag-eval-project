from langchain_community.document_loaders import TextLoader
from observability.langfuse_setup import langfuse
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# ============================================================
# Load documents
# ============================================================

loader = TextLoader("data/company_policy.txt")
docs = loader.load()

# ============================================================
# Split documents
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

# ============================================================
# Embeddings + Vector Store
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = db.as_retriever()

# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# ============================================================
# Test Cases
# ============================================================

test_cases = [
    {
        "question": "How many annual leave days do employees get?",
        "expected": "25 days"
    },
    {
        "question": "What is the remote work policy?",
        "expected": "3 days per week"
    },
    {
        "question": "Who is covered by medical insurance?",
        "expected": "employees and dependents"
    },
    {
        "question": "What is the maternity leave policy?",
        "expected": "I don't know"
    }
]

# ============================================================
# Test Runner
# ============================================================

def run_test(question, expected):

    # Retrieve documents
    docs = retriever.invoke(question)

    context = "\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are an HR assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    start_time = time.time()

    response = llm.invoke(prompt)

    latency = round(
        time.time() - start_time,
        2
    )

    answer = response.content.strip()

    passed = expected.lower() in answer.lower()

    print("\n" + "=" * 60)
    print("Question :", question)
    print("Expected :", expected)
    print("Actual   :", answer)
    print("Latency  :", latency, "seconds")
    print("Result   :", "PASS" if passed else "FAIL")

    # Log evaluation
    langfuse.create_event(
        name="evaluation",
        metadata={
            "question": question,
            "expected": expected,
            "actual": answer,
            "passed": passed,
            "latency_seconds": latency
        }
    )

    return {
        "question": question,
        "expected": expected,
        "actual": answer,
        "passed": passed,
        "latency": latency
    }

# ============================================================
# Evaluation Suite
# ============================================================

with langfuse.start_as_current_observation(
    name="rag-eval-suite"
):

    print(
        "Trace ID:",
        langfuse.get_current_trace_id()
    )

    results = []

    for tc in test_cases:

        result = run_test(
            tc["question"],
            tc["expected"]
        )

        results.append(result)

# ============================================================
# Summary
# ============================================================

passed_count = sum(
    1 for r in results if r["passed"]
)

total_count = len(results)

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"Passed: {passed_count}/{total_count}")
print(
    f"Success Rate: {(passed_count / total_count) * 100:.1f}%"
)

# ============================================================
# Flush Langfuse
# ============================================================

langfuse.flush()