import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from wxo_agentic_evaluation.metrics.metrics import Metric
from wxo_agentic_evaluation.prompt.template_render import LLMaaJTemplateRenderer
from wxo_agentic_evaluation.service_provider.provider import Provider
from wxo_agentic_evaluation.type import EvaluationData, Message
from wxo_agentic_evaluation.utils.messages_parser import ParsedMessages

root_dir: str = os.path.dirname(os.path.dirname(__file__))
LLMAAJ_PROMPT_PATH = os.path.join(root_dir, "prompt", "llmaaj_prompt.jinja2")


class Extractor(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for the extractor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def extract(
        messages: list[Message],
        **kwargs,
    ) -> Any:
        """Extract data from messages."""
        raise NotImplementedError


class Evaluation(ABC):
    """Abstract base class for all evaluations."""

    def __init__(self, llm_client: Optional[Provider] = None) -> None:
        self._llm_client = llm_client

    @property
    def llm_client(self) -> Any:
        """Access client, require it if used."""
        if self._llm_client is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires a client, but none was provided"
            )
        return self._llm_client

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for the evaluator."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        messages: list[Message],
        ground_truth: EvaluationData,
        extracted_context: Dict[str, Any],
    ) -> Optional[Metric]:
        """
        Evaluation method.

        Args:
            messages: agent and user conversational messages (includes tool calls)
            ground_truth: ground truth data
            extracted_context: dictionary containing data derived from the messages

        Returns:
            Metic
        """
        raise NotImplementedError


class LLMaaJEvaluation(Evaluation, ABC):
    """Evaluation metric for LLMaaJ."""

    @property
    @abstractmethod
    def llmaaj_instructions(self) -> str:
        """LLMaaJ instructions for the evaluator."""
        raise NotImplementedError

    @abstractmethod
    def format_llm_output(self, string: str) -> int | float | bool | str:
        """Format the output of the LLMaaJ query."""
        raise NotImplementedError

    @property
    def selected_context_keys(self) -> set[str]:
        """Override to implement context keys to pass to the prompt."""
        return set()

    def select_context(
        self, extracted_context: Dict[str, Any]
    ) -> dict[str, Any]:
        """Additional context to be added to the prompt."""
        selected_context = {
            key: value
            for key, value in extracted_context.items()
            if key in self.selected_context_keys
        }

        return selected_context

    def evaluate(
        self,
        messages: list[Message],
        ground_truth: EvaluationData,
        extracted_context: Dict[str, Any],
    ) -> Optional[Metric]:
        renderer = LLMaaJTemplateRenderer(LLMAAJ_PROMPT_PATH)
        parsed = ParsedMessages(messages=messages)
        if parsed.user_input is None or parsed.agent_response is None:
            return None
        context = str(self.select_context(extracted_context))
        prompt = renderer.render(
            user_input=parsed.user_input,
            agent_answer=parsed.agent_response,
            llmaaj_instructions=self.llmaaj_instructions,
            context=context,
        )
        score_str = self.llm_client.query(prompt)
        value = self.format_llm_output(score_str)
        return Metric(eval_name=self.name, value=value)
