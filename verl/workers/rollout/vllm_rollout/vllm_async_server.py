# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
import logging
from collections.abc import AsyncGenerator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import ray
import time
import torch
import threading
import asyncio
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from verl import DataProto
from contextlib import contextmanager
from tensordict import TensorDict
from torch import nn
from vllm import SamplingParams
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase

from verl.utils.fs import copy_to_local
from verl.workers.rollout.async_server import AsyncServerBase

from uuid import uuid4
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length

logger = logging.getLogger(__file__)


class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.vllm_config.instance_id is not None, "instance_id must be set for external ray actors."

        fields = self.vllm_config.instance_id.split(":")
        assert len(fields) == 4, f"instance_id: {self.vllm_config.instance_id} must be in the format of <namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
        namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

        # Make sure subprocess in same namespace as parent actor.
        # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
        ray.init(namespace=namespace)
        actor_names = [actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")]

        vllm_tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        assert len(actor_names) == vllm_dp_size * vllm_tp_size, f"instance_id: {self.vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: {vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."

        def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
            fields = actor_name.split(":")
            assert len(fields) == 2, f"invalid actor name: {actor_name}"
            pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
            return pg_index, local_rank

        # sort actor names by pg_index and local_rank
        actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
        actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
        self.workers: List[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
        print(f"instance_id: {self.vllm_config.instance_id} intializes with external actors: {actor_names}")

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        # for method in [method for method in dir(self.workers[0]) if callable(getattr(self.workers[0], method))]:
        #     print(f"- {method}")

        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        print(f"instance_id: {self.vllm_config.instance_id} intializes finished.")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method

        outputs = ray.get([worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers])
        return outputs

    def check_health(self):
        return


@ray.remote(num_cpus=1)
class AsyncvLLMServer(AsyncServerBase):
    """
    AsyncvLLMServer is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Start FastAPI server first.
    2. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    3. AsyncLLM spawn EngineCore in subprocess.
    4. EngineCore initialize ExternalRayDistributedExecutor.
    5. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    6. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str):
        """
        Args:
            config: DictConfig, actor_rollout_ref config.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        super().__init__()

        self.config = config
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.engine: AsyncLLM = None

        # Init user provided chat scheduler in sperate thread.
        self.generation_loop = None
        self.generation_ready = threading.Event()
        self.generation_thread = threading.Thread(target=self._init_generation_loop, daemon=True)
        self.generation_thread.start()

    def _init_generation_loop(self):
        self.generation_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.generation_loop)
        self.generation_loop.run_forever()

    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"override_generation_config: {kwargs}")
        self.sampling_params = kwargs

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=ExternalRayDistributedExecutor,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=False,
            trust_remote_code=trust_remote_code,
            seed=self.vllm_dp_rank,
        )

        # init async llm engine
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"
        self.engine = AsyncLLM.from_vllm_config(vllm_config)
        # self.engine = AsyncLLMEngine.from_vllm_config(vllm_config)

        self.pad_token_id = self.engine.tokenizer.tokenizer.pad_token_id if hasattr(self.engine.tokenizer.tokenizer, 'pad_token_id') is not None \
                                else self.engine.tokenizer.tokenizer.eos_token_id
        self.eos_token_id = self.engine.tokenizer.tokenizer.eos_token_id
        self.collect_tasks = []
        self.prompt_info = {}
        self.output_buffer = {} # store request_output
        self.partial_enable_ids = []
        self.replay_buffer: dict[str, dict] = {} # for partial rollout

    async def wake_up(self):
        await self.engine.wake_up()

    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self.engine.reset_prefix_cache()
        await self.engine.sleep()

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        # self.sampling_params['skip_special_tokens'] = False
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    async def collect_output(self, output_generator, request_id, do_print=False, async_mode=False):
        final_output = None
        try:
            async for output in output_generator:
                final_output = output
                if do_print:
                    print(f"Partial result: {output.outputs[0].text}")
            if async_mode:
                if request_id in self.replay_buffer:
                    if 'token_ids' in self.replay_buffer[request_id]:
                        final_output.outputs[0].token_ids = \
                            self.replay_buffer[request_id]["token_ids"] + final_output.outputs[0].token_ids
                    else:
                        self.output_buffer[request_id] = final_output
                else:
                    self.output_buffer[request_id] = final_output
        except asyncio.CancelledError:
            await self.engine.abort(request_id)
            if request_id in self.partial_enable_ids:
                self.replay_buffer[request_id]["gen_output"] = final_output
            # print(f"async server cancel request {request_id}")
            raise
        
        if request_id in self.partial_enable_ids:
            self.partial_enable_ids.remove(request_id)
            del self.replay_buffer[request_id]
        return final_output

    def rollout_finished(self, item) -> bool:
        # item should be DataProtoItem, mainly for generate_sequences_fused
        response = item.batch['responses']

        # 1. check if eos_token exists
        eos_token_id = item.meta_info['eos_token_id']
        eos_mask = torch.isin(response, torch.tensor(eos_token_id, device=response.device)).int()
        if eos_mask.sum().item() > 0:
            return True
        
        # 2. check if length reaches max_length
        response_mask = self.get_response_mask_by_pad_id(response)
        response_length = response_mask.sum().float().item()
        if response_length >= self.config.response_length:
            return True
        
        return False
    
    def get_response_mask_by_pad_id(self, response_id: torch.Tensor, dtype=torch.int64):
        response_mask = torch.isin(response_id, torch.tensor(self.pad_token_id, device=response_id.device)).int()
        response_mask = response_mask.eq(0).to(dtype)
        return response_mask

    def _post_process_output(self, request_outputs) -> Tuple[torch.Tensor, List[bool], List[bool]]:
        output_token_ids = []
        output_finished = []
        seq_finished = []
        
        for request_output in request_outputs:  # List[RequestOutput]
            outputs = request_output.outputs
            output_finished.extend([request_output.finished for _ in outputs])
            for output in outputs:  # List[CompletionOutput], usually len == 1
                output_token_ids.append(torch.tensor(output.token_ids))
                seq_finished.append(output.finished())

        pad_token_id = self.engine.tokenizer.tokenizer.pad_token_id if hasattr(self.engine.tokenizer.tokenizer, 'pad_token_id') is not None \
                        else self.engine.tokenizer.tokenizer.eos_token_id
        output_token_ids = pad_sequence(output_token_ids, batch_first=True, padding_value=pad_token_id)
        # output_fused already repeats n
        return output_token_ids, output_finished, seq_finished

    def start_generation(self):
        self.stop_flag = False

    def stop_generation(self):
        self.stop_flag = True
        for task in self.collect_tasks:
            task.cancel()
        self.collect_tasks.clear()
        for request_id in list(self.prompt_info.keys()):
            if request_id in self.partial_enable_ids:
                continue
            else:
                del self.prompt_info[request_id]
            if request_id in self.output_buffer:
                del self.output_buffer[request_id]

    def add_generation_task(self, raw_prompt, sampling_params, request_id, do_print=False, async_mode=False):
        output_generator = self.engine.generate(
                                prompt=raw_prompt,
                                sampling_params=sampling_params,
                                request_id=request_id
                            )
        task = asyncio.create_task(
                    self.collect_output(
                        output_generator=output_generator,
                        request_id=request_id,
                        do_print=do_print,
                        async_mode=async_mode
                    )
                )
        self.collect_tasks.append(task)

    def transfer_replay(self):
        request_ids = list(self.replay_buffer.keys())
        for request_id in request_ids:
            gen_output = self.replay_buffer[request_id]["gen_output"]
            if "token_ids" in self.replay_buffer[request_id]:
                token_ids = self.replay_buffer[request_id]["token_ids"] + gen_output.outputs[0].token_ids
            else:
                token_ids = gen_output.outputs[0].token_ids
            self.replay_buffer[request_id]["token_ids"] = token_ids
            self.replay_buffer[request_id]["gen_output"] = None
            # add to generate
            self.replay_buffer[request_id]["sampling_params"]['max_tokens'] = \
                self.config.rollout.response_length - len(token_ids)
            prompt = self.replay_buffer[request_id]["raw_prompt"] + token_ids
            self.add_generation_task(
                raw_prompt=prompt,
                sampling_params=SamplingParams(**self.replay_buffer[request_id]["sampling_params"]),
                request_id=request_id,
                # do_print=(batch_index < 1)
                async_mode=True
            )
        return len(request_ids) > 0

    async def generate_sequences_async(self, prompts, **kwargs):
        if 'uid' in prompts.non_tensor_batch.keys():
            uids = prompts.non_tensor_batch['uid']
        else:
            raise ValueError("Uids of prompts is needed in generate_sequences_async")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.rollout.val_kwargs.top_k,
                "top_p": self.config.rollout.val_kwargs.top_p,
                "temperature": self.config.rollout.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            for batch_index, raw_prompt in enumerate(prompts.non_tensor_batch['raw_prompt']):
                if batch_index < 1:
                    print(f"conversation: {raw_prompt}")
                request_id = uids[batch_index]
                if prompts.meta_info['partial_rollout_enable']:
                    assert self.config.rollout.n == 1, f"when using partial rollout in async rollout, \
                                                    n must be equal to 1"
                    self.replay_buffer[request_id] = {
                        'raw_prompt': self.engine.tokenizer.encode(raw_prompt),
                        'sampling_params': self.sampling_params,
                    }
                    self.partial_enable_ids.append(request_id)
                self.add_generation_task(
                    raw_prompt=raw_prompt,
                    sampling_params=SamplingParams(**self.sampling_params),
                    request_id=request_id,
                    # do_print=(batch_index < 1)
                    async_mode=True
                )
                self.prompt_info[request_id] = prompts[batch_index]

    async def collect_outputs_async(self, batch_size: int):
        batch_size = int(batch_size)
        tasks = set(self.collect_tasks)
        while len(self.output_buffer) < batch_size and tasks:
            done, tasks = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
        if self.stop_flag:
            return None
        self.collect_tasks = list(tasks)
        # while len(self.output_buffer) < batch_size:
        #     await asyncio.sleep(0.2)
        generate_outputs = []
        selected_prompt_info = []
        selected_uids = list(self.output_buffer.keys())[:batch_size]
        # use uid to select responding outputs
        for k in selected_uids:
            generate_outputs.append(self.output_buffer.pop(k))
            selected_prompt_info.append(self.prompt_info.pop(k))
        
        idx = torch.tensor([info.batch['input_ids'].tolist() for info in selected_prompt_info]).cpu()
        attention_mask = torch.tensor([info.batch['attention_mask'].tolist() for info in selected_prompt_info]).to(idx.device)
        position_ids = torch.tensor([info.batch['position_ids'].tolist() for info in selected_prompt_info]).to(idx.device)
        do_sample = selected_prompt_info[0].meta_info.get("do_sample", True)
        output = self._post_process_output(generate_outputs)

        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        response = output[0].to(idx.device)
        # output_finished = output[1]
        # seq_finished = output[2]

        if response.shape[1] < self.config.rollout.response_length:
            response = pad_sequence_to_length(response, self.config.rollout.response_length, self.pad_token_id)

        if self.config.rollout.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.rollout.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.rollout.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.rollout.n, dim=0)
            batch_size = batch_size * self.config.rollout.n
            # output_finished = np.repeat(np.array(output_finished, dtype=object), self.config.rollout.n, axis=0)
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = self.get_response_mask_by_pad_id(response)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        batch_dict = {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            }

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            batch_dict,
            batch_size=batch_size)
        non_tensor_batch = {
            'uid': np.array(selected_uids, dtype=object)
        }

        output_proto = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        
        return output_proto

    async def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        partial_rollout_enable = False
        if 'partial_rollout_enable' in prompts.meta_info:
            partial_rollout_enable = prompts.meta_info['partial_rollout_enable']
        if 'uid' in prompts.non_tensor_batch.keys():
            uids = prompts.non_tensor_batch.pop('uid')
        else:
            uids = None

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        # eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        # idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        # for i in range(batch_size):
        #     idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.rollout.val_kwargs.top_k,
                "top_p": self.config.rollout.val_kwargs.top_p,
                "temperature": self.config.rollout.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            tasks = []
            generate_outputs = []
            for batch_index, raw_prompt in enumerate(prompts.non_tensor_batch['raw_prompt']):
                if batch_index < 1:
                    print(f"conversation: {raw_prompt}")
                if uids is None:
                    request_id = uuid4().hex
                else:
                    request_id = uids[batch_index]
                self.add_generation_task(raw_prompt=raw_prompt,
                                        sampling_params=SamplingParams(**self.sampling_params),
                                        request_id=request_id)
            generate_outputs = await asyncio.gather(*self.collect_tasks)
            # generate_outputs = await asyncio.gather(*tasks)
            self.collect_tasks.clear()

        # print(f"generate_outputs = {generate_outputs}")
        output = self._post_process_output(generate_outputs)

        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        response = output[0].to(idx.device)
        # output_finished = output[1]
        # seq_finished = output[2]

        if response.shape[1] < self.config.rollout.response_length:
            response = pad_sequence_to_length(response, self.config.rollout.response_length, self.pad_token_id)

        if partial_rollout_enable:
            n_seqs: bool = self.config.rollout.n > 1
            # self.inference_engine.reschedule_partial_requests(n_seqs)

        if self.config.rollout.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.rollout.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.rollout.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.rollout.n, dim=0)
            batch_size = batch_size * self.config.rollout.n
            # output_finished = np.repeat(np.array(output_finished, dtype=object), self.config.rollout.n, axis=0)
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        # response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        response_attention_mask = self.get_response_mask_by_pad_id(response)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        batch_dict = {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            }

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            batch_dict,
            batch_size=batch_size)

        output_proto = DataProto(batch=batch)
        
        return output_proto