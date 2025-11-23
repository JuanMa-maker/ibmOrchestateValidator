import logging
import os

from rich.console import Console
from rich.logging import RichHandler

from wxo_agentic_evaluation.arg_configs import ProviderConfig
from wxo_agentic_evaluation.service_provider.gateway_provider import (
    GatewayProvider,
)
from wxo_agentic_evaluation.service_provider.model_proxy_provider import (
    ModelProxyProvider,
)
from wxo_agentic_evaluation.service_provider.ollama_provider import (
    OllamaProvider,
)
from wxo_agentic_evaluation.service_provider.referenceless_provider_wrapper import (
    GatewayProviderLLMKitWrapper,
    ModelProxyProviderLLMKitWrapper,
    WatsonXLLMKitWrapper,
)
from wxo_agentic_evaluation.service_provider.watsonx_provider import (
    WatsonXProvider,
)

USE_GATEWAY_MODEL_PROVIDER: bool = (
    os.environ.get("USE_GATEWAY_MODEL_PROVIDER", "FALSE").upper() == "TRUE"
)


_logging_console = Console(stderr=True)

logger = logging.getLogger(__name__)


def get_log_level_from_env():

    level_env = os.getenv("WXO_EVALUATION_LOGLEVEL")
    return level_env


LOGGING_ENABLED = get_log_level_from_env() is not None


def configure_logging_for_package_from_env(
    package_name: str = "wxo_agentic_evaluation",
    ensure_output: bool = True,
) -> None:
    """
    Configure logging using the env var WXO_EVALUATION_LOGLEVEL - no logging if that's not set
    """
    try:
        level_env = get_log_level_from_env()
        if not level_env:
            return

        level = None
        upper = level_env.strip().upper()
        if hasattr(logging, upper):
            level = getattr(logging, upper, None)

        pkg_logger = logging.getLogger(package_name)
        pkg_logger.setLevel(level)

        if ensure_output:
            if not pkg_logger.handlers:
                handler = RichHandler(
                    console=_logging_console,
                    rich_tracebacks=True,
                    show_time=False,
                    show_level=False,
                    show_path=False,
                    markup=True,
                    enable_link_path=True,
                    omit_repeated_times=True,
                    tracebacks_theme="github-dark",
                )
                handler.setFormatter(
                    logging.Formatter("%(levelname)s %(message)s")
                )
                handler.setLevel(logging.NOTSET)
                pkg_logger.addHandler(handler)
            pkg_logger.propagate = False

        # Quiet common noisy debug libs
        for name in (
            "urllib3",
            "urllib3.connectionpool",
            "requests.packages.urllib3",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)
    except:
        logger.warning("Input log level %s not valid", level_env)


configure_logging_for_package_from_env()


def _instantiate_provider(
    config: ProviderConfig, is_referenceless_eval: bool = False, **kwargs
):

    if config.provider == "watsonx":
        logger.info("Instantiate watsonx provider")
        if is_referenceless_eval:
            provider = WatsonXLLMKitWrapper
        else:
            provider = WatsonXProvider
        return provider(
            model_id=config.model_id,
            embedding_model_id=config.embedding_model_id,
            **kwargs,
        )
    elif config.provider == "ollama":
        logger.info("Instantiate Ollama")
        return OllamaProvider(model_id=config.model_id, **kwargs)

    elif config.provider == "gateway":
        logger.info("Instantiate gateway inference provider")
        if is_referenceless_eval:
            provider = GatewayProviderLLMKitWrapper
        else:
            provider = GatewayProvider
        return provider(
            model_id=config.model_id,
            embedding_model_id=config.embedding_model_id,
            **kwargs,
        )

    elif config.provider == "model_proxy":
        logger.info("Instantiate model proxy provider")
        if is_referenceless_eval:
            provider = ModelProxyProviderLLMKitWrapper
        else:
            provider = ModelProxyProvider

        return provider(
            model_id=config.model_id,
            embedding_model_id=config.embedding_model_id,
            **kwargs,
        )

    else:
        raise RuntimeError(
            f"target provider is not supported {config.provider}"
        )


def get_provider(
    config: ProviderConfig = None,
    model_id: str = None,
    embedding_model_id: str = None,
    referenceless_eval: bool = False,
    **kwargs,
):

    if config:
        return _instantiate_provider(config, **kwargs)

    if not model_id:
        raise ValueError("model_id must be provided if config is not supplied")

    if USE_GATEWAY_MODEL_PROVIDER:
        logger.info("[d b]Using gateway inference provider override")
        config = ProviderConfig(provider="gateway", model_id=model_id)
        return _instantiate_provider(config, referenceless_eval, **kwargs)

    if "WATSONX_APIKEY" in os.environ and "WATSONX_SPACE_ID" in os.environ:
        logger.info("[d b]Using watsonx inference provider")
        config = ProviderConfig(
            provider="watsonx",
            model_id=model_id,
            embedding_model_id=embedding_model_id,
        )
        return _instantiate_provider(config, referenceless_eval, **kwargs)

    if "WO_INSTANCE" in os.environ:
        logger.info("[d b]Using model_proxy inference provider")
        config = ProviderConfig(provider="model_proxy", model_id=model_id)
        return _instantiate_provider(config, referenceless_eval, **kwargs)

    logger.info("[d b]Using gateway inference provider default")
    config = ProviderConfig(provider="gateway", model_id=model_id)
    return _instantiate_provider(config, referenceless_eval, **kwargs)
