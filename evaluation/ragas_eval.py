from datasets import Dataset

from ragas import evaluate

try:
    # Newer versions
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
    )
except:
    # Older versions fallback
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
    )


def evaluate_response(
    question,
    answer,
    contexts,
    ground_truth
):
    """
    Evaluate one RAG response with RAGAS.
    """

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return result.to_pandas().to_dict(
        orient="records"
    )[0]