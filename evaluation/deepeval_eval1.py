import re

from langchain_ollama import ChatOllama

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)


# ============================================================
# DeepEval Judge Model (Ollama)
# ============================================================

class OllamaDeepEvalModel(DeepEvalBaseLLM):

    def __init__(self, model_name="qwen3.5:latest"):
        self.model_name = model_name

        self.model = ChatOllama(
            model=model_name,
            temperature=0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:

        response = self.model.invoke(prompt)

        text = response.content

        # Remove Qwen reasoning blocks if present
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL
        )

        return text.strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


# ============================================================
# Judge Model Instance
# ============================================================

judge_model = OllamaDeepEvalModel()


# ============================================================
# Evaluation Function
# ============================================================

def evaluate_response(
    question,
    answer,
    expected,
    contexts
):

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=contexts
    )

    metrics = [
        AnswerRelevancyMetric(
            model=judge_model,
            threshold=0.7,
            include_reason=True,
            async_mode=False,
        ),
        FaithfulnessMetric(
            model=judge_model,
            threshold=0.7,
            include_reason=True,
            async_mode=False,
        ),
        ContextualPrecisionMetric(
            model=judge_model,
            threshold=0.7,
            include_reason=True,
            async_mode=False,
        ),
        ContextualRecallMetric(
            model=judge_model,
            threshold=0.7,
            include_reason=True,
            async_mode=False,
        ),
    ]

    results = {}

    for metric in metrics:

        metric_name = metric.__class__.__name__

        try:

            metric.measure(test_case)

            results[metric_name] = {
                "score": float(metric.score)
                if metric.score is not None
                else 0.0,
                "passed": bool(metric.success),
                "reason": metric.reason
                if metric.reason
                else ""
            }

        except Exception as e:

            results[metric_name] = {
                "score": 0.0,
                "passed": False,
                "reason": f"Metric failed: {str(e)}"
            }

    valid_scores = [
        metric_data["score"]
        for metric_data in results.values()
    ]

    overall_score = round(
        sum(valid_scores) / len(valid_scores),
        3
    )

    overall_pass = all(
        metric_data["passed"]
        for metric_data in results.values()
    )

    return {
        "overall_score": overall_score,
        "overall_pass": overall_pass,
        "metrics": results
    }