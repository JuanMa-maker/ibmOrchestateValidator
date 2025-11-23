import dataclasses
import json
import logging
import os
import time
import uuid
from threading import Lock
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import requests

from wxo_agentic_evaluation.service_provider.provider import (
    ChatResult,
    Provider,
)

logger = logging.getLogger(__name__)

# IAM
ACCESS_URL = "https://iam.cloud.ibm.com/identity/token"
ACCESS_HEADER = {
    "content-type": "application/x-www-form-urlencoded",
    "accept": "application/json",
}

YPQA_URL = "https://yp-qa.ml.cloud.ibm.com"
PROD_URL = "https://us-south.ml.cloud.ibm.com"

DEFAULT_PARAM = MappingProxyType(
    {"min_new_tokens": 1, "decoding_method": "greedy", "max_new_tokens": 400}
)


def _truncate(value: Any, max_len: int = 1000) -> str:
    if value is None:
        return ""
    s = str(value)
    return (
        s
        if len(s) <= max_len
        else s[:max_len] + f"... [truncated {len(s) - max_len} chars]"
    )


def _translate_params_to_chat(
    params: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """
    Translate legacy generation params to chat.completions params.
    """
    translated_params: Dict[str, Any] = {}

    if "max_new_tokens" in params:
        translated_params["max_tokens"] = params["max_new_tokens"]

    if params.get("decoding_method") == "greedy":
        translated_params.setdefault("temperature", 0)
        translated_params.setdefault("top_p", 1)

    passthrough = {
        "temperature",
        "top_p",
        "n",
        "stream",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "seed",
        "response_format",
    }
    for k in passthrough:
        if k in params:
            translated_params[k] = params[k]

    return translated_params


class WatsonXProvider(Provider):
    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        space_id: Optional[str] = None,
        api_endpoint: str = PROD_URL,
        url: str = ACCESS_URL,
        timeout: int = 60,
        params: Optional[Any] = None,
        embedding_model_id: Optional[str] = None,
        use_legacy_query: Optional[bool] = None,
        system_prompt: Optional[str] = None,
        token: Optional[str] = None,
        instance_url: Optional[str] = None,
    ):
        super().__init__(use_legacy_query=use_legacy_query)

        self.url = url
        if (embedding_model_id is None) and (model_id is None):
            raise Exception(
                "either model_id or embedding_model_id must be specified"
            )
        self.model_id = model_id
        logger.info("[d b]Using inference model %s", self.model_id)
        api_key = os.environ.get("WATSONX_APIKEY", api_key)
        if not api_key:
            raise Exception("apikey must be specified")
        self.api_key = api_key
        self.access_data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.api_key,
        }
        self.api_endpoint = (api_endpoint or PROD_URL).rstrip("/")
        space_id = os.environ.get("WATSONX_SPACE_ID", space_id)
        if not space_id:
            raise Exception("space id must be specified")
        self.space_id = space_id
        self.timeout = timeout
        self.embedding_model_id = embedding_model_id
        self.lock = Lock()

        self.params = params if params is not None else DEFAULT_PARAM
        if isinstance(self.params, MappingProxyType):
            self.params = dict(self.params)
        if dataclasses.is_dataclass(self.params):
            self.params = dataclasses.asdict(self.params)

        self.system_prompt = system_prompt

        self.refresh_time = None
        self.access_token = None
        self._refresh_token()

        self.LEGACY_GEN_URL = (
            f"{self.api_endpoint}/ml/v1/text/generation?version=2023-05-02"
        )
        self.CHAT_COMPLETIONS_URL = f"{self.api_endpoint}/ml/v1/text/chat"
        self.EMBEDDINGS_URL = (
            f"{self.api_endpoint}/ml/v1/text/embeddings?version=2023-10-25"
        )

    def _get_access_token(self):
        response = requests.post(
            self.url,
            headers=ACCESS_HEADER,
            data=self.access_data,
            timeout=self.timeout,
        )
        if response.status_code == 200:
            token_data = json.loads(response.text)
            token = token_data["access_token"]
            expiration = token_data["expiration"]
            expires_in = token_data["expires_in"]
            # 9 minutes before expire
            refresh_time = expiration - int(0.15 * expires_in)
            return token, refresh_time

        raise RuntimeError(
            f"try to acquire access token and get {response.status_code}"
        )

    def prepare_header(self):
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        return headers

    def _refresh_token(self):
        # if we do not have a token or the current timestamp is 9 minutes away from expire.
        if not self.access_token or time.time() > self.refresh_time:
            with self.lock:
                if not self.access_token or time.time() > self.refresh_time:
                    (
                        self.access_token,
                        self.refresh_time,
                    ) = self._get_access_token()

    def old_query(self, sentence: Union[str, Mapping[str, str]]) -> str:
        """
        Legacy /ml/v1/text/generation
        """
        if self.model_id is None:
            raise Exception("model id must be specified for text generation")

        self._refresh_token()
        headers = self.prepare_header()

        payload: Dict[str, Any] = {
            "model_id": self.model_id,
            "input": sentence,
            "parameters": self.params or {},
            "space_id": self.space_id,
        }

        request_id = str(uuid.uuid4())
        t0 = time.time()

        logger.debug(
            "[d][b]Sending text.generation request | request_id=%s url=%s model=%s space_id=%s params=%s input_preview=%s",
            request_id,
            self.LEGACY_GEN_URL,
            self.model_id,
            self.space_id,
            json.dumps(
                payload.get("parameters", {}),
                sort_keys=True,
                ensure_ascii=False,
            ),
            _truncate(sentence, 200),
        )

        resp = None
        try:
            resp = requests.post(
                url=self.LEGACY_GEN_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            duration_ms = int((time.time() - t0) * 1000)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and "results" in data and data["results"]:
                result = data["results"][0]
            elif isinstance(data, dict):
                result = data
            else:
                raise ValueError(
                    f"Unexpected response type from WatsonX: {type(data)}"
                )

            output_text = ""
            if isinstance(result, dict):
                output_text = (
                    result.get("generated_text") or result.get("message") or ""
                )

            usage = data.get("usage") or {}
            if not usage and isinstance(result, dict):
                in_tok = result.get("input_token_count")
                out_tok = result.get("generated_token_count") or result.get(
                    "output_token_count"
                )
                if in_tok is not None or out_tok is not None:
                    usage = {
                        "prompt_tokens": in_tok,
                        "completion_tokens": out_tok,
                        "total_tokens": (in_tok or 0) + (out_tok or 0),
                    }

            api_request_id = resp.headers.get(
                "x-request-id"
            ) or resp.headers.get("request-id")

            logger.debug(
                "[d][b]text.generation response received | request_id=%s status_code=%s duration_ms=%s usage=%s output_preview=%s api_request_id=%s",
                request_id,
                resp.status_code,
                duration_ms,
                json.dumps(usage, sort_keys=True, ensure_ascii=False),
                _truncate(output_text, 2000),
                api_request_id,
            )

            if output_text:
                return output_text
            raise ValueError(
                f"Unexpected response from legacy endpoint: {data}"
            )

        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            status_code = getattr(resp, "status_code", None)
            resp_text_preview = (
                _truncate(getattr(resp, "text", None), 2000)
                if resp is not None
                else None
            )

            logger.exception(
                "text.generation request failed | request_id=%s status_code=%s duration_ms=%s response_text_preview=%s",
                request_id,
                status_code,
                duration_ms,
                resp_text_preview,
            )
            with self.lock:
                if (
                    "authentication_token_expired" in str(e)
                    or status_code == 401
                ):
                    try:
                        self.access_token, self.refresh_time = (
                            self._get_access_token()
                        )
                    except Exception:
                        pass
            raise

    def new_query(self, sentence: str) -> str:
        """
        /ml/v1/text/chat
        Returns assistant content as a plain string.
        """
        if self.model_id is None:
            raise Exception("model id must be specified for text generation")

        self._refresh_token()
        headers = self.prepare_header()

        messages: List[Dict[str, Any]] = []
        if getattr(self, "system_prompt", None):
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sentence,
                    }
                ],
            }
        )

        chat_params = _translate_params_to_chat(self.params)
        if "time_limit" in self.params:
            chat_params["time_limit"] = self.params["time_limit"]

        payload: Dict[str, Any] = {
            "model_id": self.model_id,
            "space_id": self.space_id,
            "messages": messages,
            **chat_params,
        }

        url = f"{self.CHAT_COMPLETIONS_URL}?version=2024-10-08"
        request_id = str(uuid.uuid4())
        t0 = time.time()

        logger.debug(
            "[d][b]Sending chat.completions request | request_id=%s url=%s model=%s space_id=%s params=%s input_preview=%s",
            request_id,
            url,
            self.model_id,
            self.space_id,
            json.dumps(chat_params, sort_keys=True, ensure_ascii=False),
            _truncate(sentence, 200),
        )

        resp = None
        try:
            resp = requests.post(
                url=url, headers=headers, json=payload, timeout=self.timeout
            )
            duration_ms = int((time.time() - t0) * 1000)
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage", {})
            api_request_id = resp.headers.get(
                "x-request-id"
            ) or resp.headers.get("request-id")

            logger.debug(
                "[d][b]chat.completions response received | request_id=%s status_code=%s duration_ms=%s finish_reason=%s usage=%s output_preview=%s api_request_id=%s",
                request_id,
                resp.status_code,
                duration_ms,
                finish_reason,
                json.dumps(usage, sort_keys=True, ensure_ascii=False),
                _truncate(content, 2000),
                api_request_id,
            )

            return content

        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            status_code = getattr(resp, "status_code", None)
            resp_text_preview = (
                _truncate(getattr(resp, "text", None), 2000)
                if resp is not None
                else None
            )

            logger.exception(
                "chat.completions request failed | request_id=%s status_code=%s duration_ms=%s response_text_preview=%s",
                request_id,
                status_code,
                duration_ms,
                resp_text_preview,
            )
            with self.lock:
                if (
                    "authentication_token_expired" in str(e)
                    or status_code == 401
                ):
                    try:
                        self.access_token, self.refresh_time = (
                            self._get_access_token()
                        )
                    except Exception:
                        pass
            raise

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        params: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """
        Sends a multi-message chat request to /ml/v1/text/chat
        Returns ChatResult with text, usage, finish_reason, and raw response.
        """
        if self.model_id is None:
            raise Exception("model id must be specified for chat")

        self._refresh_token()
        headers = self.prepare_header()

        wx_messages: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user" and isinstance(content, str):
                wx_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": content}],
                    }
                )
            else:
                wx_messages.append({"role": role, "content": content})

        merged_params = dict(self.params or {})
        if params:
            merged_params.update(params)
        chat_params = _translate_params_to_chat(merged_params)
        chat_params.pop("stream", None)
        if "time_limit" in merged_params:
            chat_params["time_limit"] = merged_params["time_limit"]

        payload: Dict[str, Any] = {
            "model_id": self.model_id,
            "space_id": self.space_id,
            "messages": wx_messages,
            **chat_params,
        }

        url = f"{self.CHAT_COMPLETIONS_URL}?version=2024-10-08"
        request_id = str(uuid.uuid4())
        t0 = time.time()

        last_user = next(
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        logger.debug(
            "[d][b]Sending chat.completions request (non-streaming) | request_id=%s url=%s model=%s space_id=%s params=%s input_preview=%s",
            request_id,
            url,
            self.model_id,
            self.space_id,
            json.dumps(chat_params, sort_keys=True, ensure_ascii=False),
            _truncate(last_user, 200),
        )

        resp = None
        try:
            resp = requests.post(
                url=url, headers=headers, json=payload, timeout=self.timeout
            )
            duration_ms = int((time.time() - t0) * 1000)
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage", {})
            api_request_id = resp.headers.get(
                "x-request-id"
            ) or resp.headers.get("request-id")

            logger.debug(
                "[d][b]chat.completions response received (non-streaming) | request_id=%s status_code=%s duration_ms=%s finish_reason=%s usage=%s output_preview=%s api_request_id=%s",
                request_id,
                resp.status_code,
                duration_ms,
                finish_reason,
                json.dumps(usage, sort_keys=True, ensure_ascii=False),
                _truncate(content, 2000),
                api_request_id,
            )

            return ChatResult(
                text=content, usage=usage, finish_reason=finish_reason, raw=data
            )

        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            status_code = getattr(resp, "status_code", None)
            resp_text_preview = (
                _truncate(getattr(resp, "text", None), 2000)
                if resp is not None
                else None
            )

            logger.exception(
                "chat.completions request failed (non-streaming) | request_id=%s status_code=%s duration_ms=%s response_text_preview=%s",
                request_id,
                status_code,
                duration_ms,
                resp_text_preview,
            )
            with self.lock:
                if (
                    "authentication_token_expired" in str(e)
                    or status_code == 401
                ):
                    try:
                        self.access_token, self.refresh_time = (
                            self._get_access_token()
                        )
                    except Exception:
                        pass
            raise

    def encode(self, sentences: List[str]) -> List[list]:
        if self.embedding_model_id is None:
            raise Exception(
                "embedding model id must be specified for text encoding"
            )

        self._refresh_token()
        headers = self.prepare_header()

        # Minimal logging for embeddings
        request_id = str(uuid.uuid4())
        t0 = time.time()
        logger.debug(
            "[d][b]Sending embeddings request | request_id=%s url=%s model=%s space_id=%s num_inputs=%s",
            request_id,
            self.EMBEDDINGS_URL,
            self.embedding_model_id,
            self.space_id,
            len(sentences),
        )

        payload = {
            "inputs": sentences,
            "model_id": self.embedding_model_id,
            "space_id": self.space_id,
        }

        resp = requests.post(
            url=self.EMBEDDINGS_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        duration_ms = int((time.time() - t0) * 1000)

        if resp.status_code == 200:
            data = resp.json()
            vectors = [entry["embedding"] for entry in data["results"]]
            logger.debug(
                "[d][b]Embeddings response received | request_id=%s status_code=%s duration_ms=%s num_vectors=%s",
                request_id,
                resp.status_code,
                duration_ms,
                len(vectors),
            )
            return vectors

        logger.error(
            "[d b red]Embeddings request failed | request_id=%s status_code=%s duration_ms=%s response_text_preview=%s",
            request_id,
            resp.status_code,
            duration_ms,
            _truncate(resp.text, 2000),
        )
        resp.raise_for_status()


if __name__ == "__main__":

    provider = WatsonXProvider(
        model_id="meta-llama/llama-3-2-90b-vision-instruct",
        use_legacy_query=False,  # set True to use legacy endpoint
        system_prompt="You are a helpful assistant.",
    )

    prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>


Your username is nwaters and you want to find out timeoff schedule for yourself for 20250101 to 20250303
<|eot_id|><|start_header_id|>user<|end_header_id|>


Reminder:
- try to respond only once per input
- if you get everything you need. respond with END

<|eot_id|>
<|start_header_id|>user<|end_header_id|>
my username is nwaters. what's my timeoff schedule?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
we need to call get_assignment_id for username nwaters. do you want to make the function call? yes/no<|eot_id|>
<|start_header_id|>user<|end_header_id|>
yes<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
ok, i have your assignment id. what's the start and end date?<|eot_id|>
<|start_header_id|>user<|end_header_id|>
start and end is 20250101 to 20250303<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Usernwaters did not take anytime off during the period<|eot_id|>

{% endfor -%}
<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

    print(provider.query(prompt))
