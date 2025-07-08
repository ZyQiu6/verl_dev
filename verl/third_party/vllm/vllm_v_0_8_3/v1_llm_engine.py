# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, Union

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.engine.metrics_types import StatLoggerBase
from vllm.inputs import PromptType
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.transformers_utils.tokenizer_group import (
    BaseTokenizerGroup, init_tokenizer_from_configs)
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor

from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine


class LLMEngine(V1LLMEngine):
    """Legacy LLMEngine for backwards compatibility."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[dict[str, StatLoggerBase]] = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
    ) -> None:
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 LLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "LLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # important: init dp group before init the engine_core
        # In the decoupled engine case this is handled in EngineCoreProc.
        parallel_config = vllm_config.parallel_config
        if not multiprocess_mode and parallel_config.data_parallel_size > 1:
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None
        self.should_execute_dummy_batch = False

        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)
        self.tokenizer.ping()

        # Processor (convert Inputs --> EngineCoreRequests)
        self.processor = Processor(vllm_config=vllm_config,
                                   tokenizer=self.tokenizer,
                                   mm_registry=mm_registry)

        # OutputProcessor (convert EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(self.tokenizer,
                                                log_stats=False)

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=False,  # FIXME: implement
        )

        if not multiprocess_mode:
            # for v0 compatibility
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

    def set_partial_rollout_save_steps(self, partial_rollout_save_steps: int) -> None:
        pass

    def set_partial_rollout_enable(self, partial_rollout_enable: bool, virtual_engine=0) -> None:
        pass
        
    def clear_rollout_steps(self, virtual_engine=0) -> None:
        pass

    def transfer_partial_to_waiting(self, virtual_engine=0) -> None:
        pass
        
    def transfer_partial_to_swapped(self, virtual_engine=0) -> None:
        pass
        
    def transfer_partial_to_running(self, virtual_engine=0) -> None:
        pass
        
    def sorted_partial_seq_groups(self, virtual_engine=0):
        pass
    
    def add_decomposed_partial_seq_group(self, seq_group, virtual_engine=0) -> None:
        pass

    def set_partial_rollout_mode(self, partial_rollout_mode: Optional[str], virtual_engine=0) -> None:
        pass
        
    def set_fuse_enable(self, fuse_enable: bool) -> None:
        pass
        
    def is_request_fused(self, request_id: str, virtual_engine=0) -> bool:
        pass
    
    def add_seq_group(self, seq_group, virtual_engine=0) -> None:
        pass

    def set_max_response_len(self, max_response_len: int) -> None:
        self.max_response_len = max_response_len
        
    def get_request_num(self, virtual_engine=0) -> int:
        pass

    def set_init_request_num(self, init_request_num: int) -> None:
        self.init_request_num = init_request_num