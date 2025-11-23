from __future__ import annotations

import logging
import os
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

from wxo_agentic_evaluation.type import ProviderInstancesCacheKey


class SingletonProviderMeta(type):

    _provider_instances: Dict[str, "Provider"] = {}
    _instantiation_lock = Lock()

    def __call__(cls, *args, **kwargs):

        key_str: str = str(cls._get_key(cls.__name__, args, kwargs))

        if key_str not in cls._provider_instances:
            with cls._instantiation_lock:
                if key_str not in cls._provider_instances:
                    cls._provider_instances[key_str] = super().__call__(
                        *args, **kwargs
                    )

        return cls._provider_instances[key_str]

    @staticmethod
    def _get_key(
        provider: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> ProviderInstancesCacheKey:

        args_str = str(args) if args else "noargs"
        kwargs_str = str(sorted(kwargs.items())) if kwargs else "nokwargs"

        return ProviderInstancesCacheKey(
            provider=provider,
            hashed_args=args_str,
            hashed_kwargs=kwargs_str,
        )


class SingletonProviderABCMeta(ABCMeta, SingletonProviderMeta):
    pass


@dataclass
class ChatResult:
    text: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    raw: Optional[Any] = None


class Provider(ABC, metaclass=SingletonProviderABCMeta):
    def __init__(
        self,
        use_legacy_query: Optional[bool] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        env_use_legacy = os.environ.get("USE_LEGACY_QUERY")
        if env_use_legacy is not None:
            self.use_legacy_query: bool = env_use_legacy.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            self.use_legacy_query = (
                bool(use_legacy_query) if use_legacy_query is not None else True
            )
        if self.use_legacy_query:
            self.logger.debug("[d][b]Using legacy /text/generation queries")
        else:
            self.logger.debug("[d][b]Using new /chat/completions queries")

    @abstractmethod
    def old_query(self, sentence: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def new_query(self, sentence: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def encode(self, sentences: List[str]) -> List[list]:
        raise NotImplementedError

    def query(self, sentence: str) -> str:
        if self.use_legacy_query:
            return self.old_query(sentence)
        return self.new_query(sentence)

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        params: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement chat()."
        )

    def batch_query(
        self,
        sentences: List[str],
        max_workers: Optional[int] = None,
    ) -> List[str]:
        if not sentences:
            return []

        if not max_workers or max_workers <= 1:
            return [self.query(sentence) for sentence in sentences]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List[Optional[str]] = [None] * len(sentences)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self.query, s): i for i, s in enumerate(sentences)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()

        return [r if r is not None else "" for r in results]

    def set_routing(self, use_legacy_query: Optional[bool] = None) -> None:
        if use_legacy_query is not None:
            self.use_legacy_query = bool(use_legacy_query)

    def close(self) -> None:
        return
