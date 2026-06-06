from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from langchain_ollama import ChatOllama


class OllamaDeepEvalModel(DeepEvalBaseLLM):

    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.model = ChatOllama(
            model=model_name,
            temperature=0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


# Create judge AFTER class definition
judge_model = OllamaDeepEvalModel()


def evaluate_response(
    question,
    answer,
    expected
):
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected
    )

    metric = AnswerRelevancyMetric(
        model=judge_model,
        threshold=0.7,
        async_mode=False,
        include_reason=True
    )

    metric.measure(test_case)

    return {
        "score": metric.score,
        "passed": metric.success,
        "reason": metric.reason
    }