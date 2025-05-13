# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py

import time
from functools import partial
from typing import Callable, Dict, Iterable, Optional, Type, Union, List

import torch.nn as nn
from vllm.config import (
    CacheConfig,
    DecodingConfig,
    DeviceConfig,
    EngineConfig,
    LoRAConfig,
    ObservabilityConfig,
    ParallelConfig,
    PromptAdapterConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.core.scheduler import Scheduler
from vllm.engine.llm_engine import LLMEngine, SchedulerContext, SchedulerOutputState, _load_generation_config_dict
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.inputs.preprocess import InputPreprocessor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sampling_params import RequestOutputKind
from vllm.engine.output_processor.util import create_output_by_sequence_group
from vllm.outputs import (EmbeddingRequestOutput, RequestOutput,
                          RequestOutputFactory)
from vllm.logger import init_logger
from vllm.sequence import (EmbeddingSequenceGroupOutput, ExecuteModelRequest,
                           Sequence, SequenceGroup, SequenceGroupMetadata,
                           SequenceStatus)
from vllm.tracing import init_tracer
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.usage.usage_lib import UsageContext, is_usage_stats_enabled, usage_message
from vllm.utils import Counter, weak_bind
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.version import __version__ as VLLM_VERSION

from .scheduler import Scheduler

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


class LLMEngine(LLMEngine):
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The :class:`~vllm.LLM` class wraps this class for offline batched inference
    and the :class:`AsyncLLMEngine` class wraps this class for online serving.

    The config arguments are derived from :class:`~vllm.EngineArgs`. (See
    :ref:`engine_args`)

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        device_config: The configuration related to the device.
        lora_config (Optional): The configuration related to serving multi-LoRA.
        speculative_config (Optional): The configuration related to speculative
            decoding.
        executor_class: The model executor class for managing distributed
            execution.
        prompt_adapter_config (Optional): The configuration related to serving
            prompt adapters.
        log_stats: Whether to log statistics.
        usage_context: Specified entry point, used for usage info collection.
    """
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        partial_rollout_save_steps: Optional[int] = None,
    ) -> None:
        if envs.VLLM_USE_V1:
            raise ValueError(
                "Using V0 LLMEngine, but envs.VLLM_USE_V1=True. "
                "This should not happen. As a workaround, try using "
                "LLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config  # noqa
        self.load_config = vllm_config.load_config
        self.decoding_config = vllm_config.decoding_config or DecodingConfig(  # noqa
        )
        self.prompt_adapter_config = vllm_config.prompt_adapter_config  # noqa
        self.observability_config = vllm_config.observability_config or ObservabilityConfig(  # noqa
        )

        logger.info(
            "Initializing a V0 LLM engine (v%s) with config: %s, "
            "use_cached_outputs=%s, ",
            VLLM_VERSION,
            vllm_config,
            use_cached_outputs,
        )

        self.log_stats = log_stats
        self.use_cached_outputs = use_cached_outputs

        if not self.model_config.skip_tokenizer_init:
            self.tokenizer = self._init_tokenizer()
            self.detokenizer = Detokenizer(self.tokenizer)
            tokenizer_group = self.get_tokenizer_group()
        else:
            self.tokenizer = None
            self.detokenizer = None
            tokenizer_group = None

        # Ensure that the function doesn't contain a reference to self,
        # to avoid engine GC issues
        def get_tokenizer_for_seq(sequence: Sequence) -> AnyTokenizer:
            assert tokenizer_group, ("tokenizer_group cannot be None, "
                                     "make sure skip_tokenizer_init is False")
            return tokenizer_group.get_lora_tokenizer(sequence.lora_request)

        self.seq_counter = Counter()
        self.generation_config_fields = (
            self.model_config.try_get_generation_config())

        self.input_preprocessor = InputPreprocessor(self.model_config,
                                                    self.tokenizer,
                                                    mm_registry)

        self.input_registry = input_registry
        self.input_processor = input_registry.create_input_processor(
            self.model_config)

        self.model_executor = executor_class(vllm_config=vllm_config, )

        if self.model_config.runner_type != "pooling":
            self._initialize_kv_caches()

        # If usage stat is enabled, collect relevant info.
        if is_usage_stats_enabled():
            from vllm.model_executor.model_loader import (
                get_architecture_class_name)
            usage_message.report_usage(
                get_architecture_class_name(self.model_config),
                usage_context,
                extra_kvs={
                    # Common configuration
                    "dtype":
                    str(self.model_config.dtype),
                    "tensor_parallel_size":
                    self.parallel_config.tensor_parallel_size,
                    "block_size":
                    self.cache_config.block_size,
                    "gpu_memory_utilization":
                    self.cache_config.gpu_memory_utilization,

                    # Quantization
                    "quantization":
                    self.model_config.quantization,
                    "kv_cache_dtype":
                    str(self.cache_config.cache_dtype),

                    # Feature flags
                    "enable_lora":
                    bool(self.lora_config),
                    "enable_prompt_adapter":
                    bool(self.prompt_adapter_config),
                    "enable_prefix_caching":
                    self.cache_config.enable_prefix_caching,
                    "enforce_eager":
                    self.model_config.enforce_eager,
                    "disable_custom_all_reduce":
                    self.parallel_config.disable_custom_all_reduce,
                })

        if self.tokenizer:
            # Ping the tokenizer to ensure liveness if it runs in a
            # different process.
            self.tokenizer.ping()

        self.cached_scheduler_outputs = [
            SchedulerOutputState()
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        self.scheduler_contexts = [
            SchedulerContext(multi_step_stream_outputs=self.scheduler_config.
                             multi_step_stream_outputs)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        if self.model_config.use_async_output_proc:
            process_model_outputs = weak_bind(self._process_model_outputs)

            self.async_callbacks = [
                partial(process_model_outputs,
                        ctx=self.scheduler_contexts[v_id])
                for v_id in range(self.parallel_config.pipeline_parallel_size)
            ]
        else:
            self.async_callbacks = []

        # Currently used by AsyncLLMEngine to ensure quick append
        # of request outputs to asyncio queues
        self.process_request_outputs_callback: Optional[Callable] = None

        # Create the scheduler.
        # NOTE: the cache_config here have been updated with the numbers of
        # GPU and CPU blocks, which are profiled in the distributed executor.
        if isinstance(self.vllm_config.scheduler_config.scheduler_cls, str):
            Scheduler = resolve_obj_by_qualname(
                self.vllm_config.scheduler_config.scheduler_cls)
        else:
            Scheduler = self.vllm_config.scheduler_config.scheduler_cls
        self.scheduler = [
            Scheduler(
                self.scheduler_config, self.cache_config, self.lora_config,
                self.parallel_config.pipeline_parallel_size,
                self.async_callbacks[v_id]
                if self.model_config.use_async_output_proc else None)
            for v_id in range(self.parallel_config.pipeline_parallel_size)
        ]

        # Metric Logging.
        if self.log_stats:
            if stat_loggers is not None:
                self.stat_loggers = stat_loggers
            else:
                # Lazy import for prometheus multiprocessing.
                # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
                # before prometheus_client is imported.
                # See https://prometheus.github.io/client_python/multiprocess/
                from vllm.engine.metrics import (LoggingStatLogger,
                                                 PrometheusStatLogger)

                self.stat_loggers = {
                    "logging":
                    LoggingStatLogger(
                        local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                        vllm_config=vllm_config),
                    "prometheus":
                    PrometheusStatLogger(
                        local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                        labels=dict(
                            model_name=self.model_config.served_model_name),
                        vllm_config=vllm_config),
                }
                self.stat_loggers["prometheus"].info("cache_config",
                                                     self.cache_config)

        self.tracer = None
        if self.observability_config.otlp_traces_endpoint:
            self.tracer = init_tracer(
                "vllm.llm_engine",
                self.observability_config.otlp_traces_endpoint)

        # Create sequence output processor, e.g. for beam search or
        # speculative decoding.
        self.output_processor = (
            SequenceGroupOutputProcessor.create_output_processor(
                self.scheduler_config,
                self.detokenizer,
                self.scheduler,
                self.seq_counter,
                get_tokenizer_for_seq,
                stop_checker=StopChecker(
                    self.scheduler_config.max_model_len,
                    get_tokenizer_for_seq,
                ),
            ))

        self.seq_id_to_seq_group: Dict[str, SequenceGroupBase] = {}

        # Flag to set when an input fails to process and the engine should run
        # the next step without re-scheduling.
        self._skip_scheduling_next_step = False
        
        # Partial rollout
        self.partial_rollout_save_steps = partial_rollout_save_steps
        self.partial_rollout_mode = None
        self.partial_rollout_enable = False
        # Overlapping
        self.fuse_enable = False
        self.max_response_len = 0
        
    def set_partial_rollout_enable(self, partial_rollout_enable: bool, virtual_engine=0) -> None:
        self.partial_rollout_enable = partial_rollout_enable
        self.scheduler[virtual_engine].set_partial_rollout_enable(partial_rollout_enable)
        
    def clear_rollout_steps(self, virtual_engine=0) -> None:
        self.scheduler[virtual_engine].clear_rollout_steps()
                    
    def step(self) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        """Performs one decoding iteration and returns newly generated results.

        .. figure:: https://i.imgur.com/sv2HssD.png
            :alt: Overview of the step function
            :align: center

            Overview of the step function.

        Details:
            - Step 1: Schedules the sequences to be executed in the next
              iteration and the token blocks to be swapped in/out/copy.

                - Depending on the scheduling policy,
                  sequences may be `preempted/reordered`.
                - A Sequence Group (SG) refer to a group of sequences
                  that are generated from the same prompt.

            - Step 2: Calls the distributed executor to execute the model.
            - Step 3: Processes the model output. This mainly includes:

                - Decodes the relevant outputs.
                - Updates the scheduled sequence groups with model outputs
                  based on its `sampling parameters` (`use_beam_search` or not).
                - Frees the finished sequence groups.

            - Finally, it creates and returns the newly generated results.

        Example:
            >>> # Please see the example/ folder for more detailed examples.
            >>>
            >>> # initialize engine and request arguments
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> example_inputs = [(0, "What is LLM?",
            >>>    SamplingParams(temperature=0.0))]
            >>>
            >>> # Start the engine with an event loop
            >>> while True:
            >>>     if example_inputs:
            >>>         req_id, prompt, sampling_params = example_inputs.pop(0)
            >>>         engine.add_request(str(req_id),prompt,sampling_params)
            >>>
            >>>     # continue the request processing
            >>>     request_outputs = engine.step()
            >>>     for request_output in request_outputs:
            >>>         if request_output.finished:
            >>>             # return or show the request output
            >>>
            >>>     if not (engine.has_unfinished_requests() or example_inputs):
            >>>         break
        """
        if self.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is only supported through AsyncLLMEngine "
                "as performance will be severely degraded otherwise.")

        # For llm_engine, there is no pipeline parallel support, so the engine
        # used is always 0.
        virtual_engine = 0

        # These are cached outputs from previous iterations. None if on first
        # iteration
        cached_outputs = self.cached_scheduler_outputs[virtual_engine]
        seq_group_metadata_list = cached_outputs.seq_group_metadata_list
        scheduler_outputs = cached_outputs.scheduler_outputs
        allow_async_output_proc = cached_outputs.allow_async_output_proc

        ctx = self.scheduler_contexts[virtual_engine]

        # Clear outputs for each new scheduler iteration
        ctx.request_outputs.clear()

        # Skip the scheduler if there are any remaining steps in the seq groups.
        # This ensures that the scheduler is only called again when the current
        # batch has completed.
        if not self._has_remaining_steps(seq_group_metadata_list):
            # Schedule iteration
            (seq_group_metadata_list, scheduler_outputs,
             allow_async_output_proc
             ) = self.scheduler[virtual_engine].schedule()
            
            if self.partial_rollout_enable and self.partial_rollout_mode == "reuse":
                scheduler_outputs.blocks_to_swap_out.extend(
                    self.scheduler[virtual_engine].get_and_reset_partial_rollout_blocks_to_swap_out()
                )
            
            ctx.seq_group_metadata_list = seq_group_metadata_list
            ctx.scheduler_outputs = scheduler_outputs

            # Maybe switch from async mode to sync mode
            if not allow_async_output_proc and len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)

            if (self.scheduler_config.is_multi_step
                    and scheduler_outputs.num_lookahead_slots > 0):
                # cache the scheduler outputs for the next iteration if we have
                # lookahead slots
                self._cache_scheduler_outputs_for_multi_step(
                    virtual_engine, seq_group_metadata_list, scheduler_outputs,
                    allow_async_output_proc)

        assert seq_group_metadata_list is not None
        assert scheduler_outputs is not None
        
        if self.partial_rollout_enable or self.fuse_enable:
            for i in range(len(ctx.seq_group_metadata_list)):
                ctx.seq_group_metadata_list[i].sampling_params.output_kind = RequestOutputKind.CUMULATIVE

        if not scheduler_outputs.is_empty():
            finished_requests_ids = self.scheduler[
                virtual_engine].get_and_reset_finished_requests_ids()

            # Check if we have a cached last_output from the previous iteration.
            # For supporting PP this is probably the best way to pass the
            # sampled_token_ids, as a separate broadcast over all the PP stages
            # will cause one virtual engine's microbatch to block the pipeline.
            last_sampled_token_ids = \
                self._get_last_sampled_token_ids(virtual_engine)

            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids)

            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[
                    virtual_engine]

            outputs = self.model_executor.execute_model(
                execute_model_req=execute_model_req)

            # We need to do this here so that last step's sampled_token_ids can
            # be passed to the next iteration for PP.
            if self.scheduler_config.is_multi_step:
                self._update_cached_scheduler_output(virtual_engine, outputs)
        else:
            # Nothing scheduled => If there is pending async postprocessor,
            # then finish it here.
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            # No outputs in this case
            outputs = []

        # Finish the current step for all the sequence groups.
        if self.scheduler_config.is_multi_step:
            for seq_group in seq_group_metadata_list:
                seq_group.finish_step()

        if not self._has_remaining_steps(seq_group_metadata_list):
            # clear the cache if we have finished all the steps.
            if self.scheduler_config.is_multi_step:
                self.cached_scheduler_outputs[0] = SchedulerOutputState()

            # is_first_step_output is True only when the num_steps of all
            # the sequences are 1. When the num_steps > 1,
            # multi_step_model_runner does the first-step output append.
            is_first_step_output: bool = False if not seq_group_metadata_list \
                else seq_group_metadata_list[0].state.num_steps == 1

            # Add results to the output_queue
            ctx.append_output(outputs=outputs,
                              seq_group_metadata_list=seq_group_metadata_list,
                              scheduler_outputs=scheduler_outputs,
                              is_async=allow_async_output_proc,
                              is_last_step=True,
                              is_first_step_output=is_first_step_output)

            if outputs and allow_async_output_proc:
                assert len(outputs) == 1, (
                    "Async postprocessor expects only a single output set")

                self._advance_to_next_step(
                    outputs[0], seq_group_metadata_list,
                    scheduler_outputs.scheduled_seq_groups)

            # Check if need to run the usual non-async path
            if not allow_async_output_proc:
                self._process_model_outputs(ctx=ctx)

                # Log stats.
                self.do_log_stats(scheduler_outputs, outputs)

                # Tracing
                self.do_tracing(scheduler_outputs)
        else:
            # Multi-step case
            return ctx.request_outputs

        for seq_group_metadata in seq_group_metadata_list:
            request_id = seq_group_metadata.request_id
            self.scheduler[virtual_engine].add_rollout_steps(request_id)
        
        if self.partial_rollout_enable:
            for seq_group_metadata in seq_group_metadata_list:
                request_id = seq_group_metadata.request_id
                if self.scheduler[virtual_engine].get_rollout_steps(request_id) > self.partial_rollout_save_steps:
                    self.scheduler[virtual_engine].transfer_partial_rollout_requests(request_id)
        
        if self.fuse_enable:
            fuse_threshold = self.partial_rollout_save_steps - 100 if self.partial_rollout_save_steps else 0.6 * self.max_response_len
            for seq_group_metadata in seq_group_metadata_list:
                request_id = seq_group_metadata.request_id
                if self.scheduler[virtual_engine].get_rollout_steps(request_id) > fuse_threshold:
                    self.scheduler[virtual_engine].add_fused_request_id(request_id)
                    self.scheduler[virtual_engine].transfer_fused_requests(request_id)
        
        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

            # Stop the execute model loop in parallel workers until there are
            # more requests to process. This avoids waiting indefinitely in
            # torch.distributed ops which may otherwise timeout, and unblocks
            # the RPC thread in the workers so that they can process any other
            # queued control plane messages, such as add/remove lora adapters.
            logger.debug("Stopping remote worker execution loop.")
            self.model_executor.stop_remote_worker_execution_loop()
        
        if self.partial_rollout_enable:
            assert len(ctx.request_outputs) == len(seq_group_metadata_list), \
            f"len(ctx.request_outputs): {len(ctx.request_outputs)},len(seq_group_metadata_list): {len(seq_group_metadata_list)}"

        return ctx.request_outputs

    def transfer_partial_to_waiting(self, virtual_engine=0) -> None:
        self.scheduler[virtual_engine].transfer_partial_to_waiting()
        
    def transfer_partial_to_swapped(self, virtual_engine=0) -> None:
        self.scheduler[virtual_engine].transfer_partial_to_swapped()
        
    def transfer_partial_to_running(self, virtual_engine=0) -> None:
        self.scheduler[virtual_engine].transfer_partial_to_running()
        
    def sorted_partial_seq_groups(self, virtual_engine=0) -> List[SequenceGroup]:
        return self.scheduler[virtual_engine].sorted_partial_seq_groups()
    
    def add_decomposed_partial_seq_group(self, seq_group, virtual_engine=0) -> None:
        self.scheduler[virtual_engine].add_partial_seq_group(seq_group)

    def set_partial_rollout_mode(self, partial_rollout_mode: Optional[str], virtual_engine=0) -> None:
        self.partial_rollout_mode = partial_rollout_mode
        self.scheduler[virtual_engine].set_partial_rollout_mode(partial_rollout_mode)
        
    def set_fuse_enable(self, fuse_enable: bool) -> None:
        self.fuse_enable = fuse_enable
        
    def is_request_fused(self, request_id: str, virtual_engine=0) -> bool:
        return self.scheduler[virtual_engine].is_request_fused(request_id)
    
    def add_seq_group(self, seq_group, virtual_engine=0) -> None:
        self.scheduler[virtual_engine].add_seq_group(seq_group)

    def set_max_response_len(self, max_response_len: int) -> None:
        self.max_response_len = max_response_len
