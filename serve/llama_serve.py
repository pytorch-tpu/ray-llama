# Copyright 2023 The Google Cloud ML Accelerators Authors. All Rights Reserved.
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
# ==============================================================================
"""Ray Serve Llama example."""

from typing import Iterable, List
from fastapi import FastAPI
from fastapi.responses import Response
import ray
from ray import serve
from ray.serve.gradio_integrations import GradioIngress

import gradio as gr
import asyncio


_VALID_MODELS = [
    "llama-2-7b",
    "llama-2-13b",
    "llama-2-70b",
]
_MAX_BATCH_SIZE = 1

app = FastAPI()

RUNTIME = {
    "env_vars": {"PJRT_DEVICE": "TPU"}
}

# Replace this with a GCS path
_CHECKPOINT_PATH = "gs://ray-llama-demo"


@serve.deployment
class MyGradioServer(GradioIngress):

    def __call__(self, text):
        generated_list = self.generator(
            text, do_sample=True, min_length=20, max_length=100
        )
        generated = generated_list[0]["generated_text"]
        return generated

    def __init__(self, llama_handle):
        self._handle = llama_handle

        super().__init__(lambda: gr.Interface(
            fn=self.fanout,
            inputs="textbox",
            outputs="textbox"))

    async def fanout(self, text: str):
        ref = await asyncio.gather(self._handle.remote(text))
        result = ray.get(ref)
        return (
            f"[GradIO]: {result}"
        )


@ray.remote(resources={"TPU": 4})
class LlamaTpuActor:
    """A LLaMA2 actor that lives on a single TPU VM actor."""
    def __init__(self,
        model_name: str,
        worker_id: int,
        tokenizer_path: str = "/tokenizer.model",
        max_batch_size: int = _MAX_BATCH_SIZE,
        max_seq_len: int = 2048,
        max_gen_len: int = 20,
        temperature: float = 0.6,
        top_p: int = 1,
        dynamo: bool = True):
        # Note - we intentionally separate the ML framework
        # initialization to another function that we can
        # `ray.get()`.

        # This is a best practice that will help us catch and
        # raise errors very quickly.
        import os
        import socket
        print(f"Using model: {model_name}.")
        self._host_name = socket.gethostname()
        self._model_name = model_name
        self._tokenizer_path = tokenizer_path
        self._max_batch_size = max_batch_size
        self._max_seq_len = max_seq_len
        self._max_gen_len = max_gen_len
        self._temperature = temperature
        self._top_p = top_p
        self._dynamo = dynamo
        self._ckpt_dir = os.path.join(_CHECKPOINT_PATH, model_name)
        self._worker_id = worker_id

    def __repr__(self) -> str:
        """Returns the actor logger prefix."""
        return f"LLaMAActor{self._model_name}::{self._host_name}"

    def initialize(self):
        """Initializes the LLaMA generator."""
        import torch
        import torch_xla
        import torch_xla.runtime as xr
        from llama import Llama
        self.generator = Llama.build(
            ckpt_dir=self._ckpt_dir,
            tokenizer_path=self._tokenizer_path,
            max_seq_len=self._max_seq_len,
            max_batch_size=self._max_batch_size,
            dynamo=self._dynamo)

    def generate(self, inputs: Iterable[str]) -> List[str]:
        print("Generating results for inputs: ", inputs)
        import torch
        with torch.no_grad():
            results = self.generator.text_completion(
                inputs,
                max_gen_len=self._max_gen_len,
                temperature=self._temperature,
                top_p=self._top_p,
            )
            return results


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,
    })
class LlamaServer:
    """A ray actor representing a shard of the Llama serving workload."""
    def __init__(
        self,
        model_name: str,
        tokenizer_path: str = "/tokenizer.model",
        max_batch_size: int = _MAX_BATCH_SIZE,
        max_seq_len: int = 2048,
        max_gen_len: int = 20,
        temperature: float = 0.6,
        top_p: int = 1,
        dynamo: bool = True):
        assert model_name in _VALID_MODELS
        tpu_pod_name = ray.util.accelerators.tpu.get_current_pod_name()
        num_tpu_pod_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
        assert ray.available_resources()[tpu_pod_name] == num_tpu_pod_hosts
        actor_def = LlamaTpuActor.options(resources={tpu_pod_name: 1, "TPU": 4})
        print("Creating TPU VM shards.")
        self._shards = [actor_def.remote(
            model_name=model_name,
            worker_id=i,
            tokenizer_path=tokenizer_path,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            dynamo=dynamo,
        ) for i in range(num_tpu_pod_hosts)]
        try:
            ray.get([s.initialize.remote() for s in self._shards])
            # warmup
            self.generate_batch(["I believe the meaning of life is ..."])
        except Exception as e:
            print("Caught error ", e)
            raise e

    def generate_batch(self, prompts: Iterable[str]) -> List[str]:
        print("Preprocessing prompts: ", prompts)
        try:
            all_results = ray.get([
                s.generate.remote(prompts) for s in self._shards])
            return all_results
        except Exception as e:
            print("Failed with ", e)

    def __repr__(self):
        return f"[{self._model_name}-shard]: "


llama_7b = LlamaServer.options(ray_actor_options={"TPU-v4-8-head": 1}).bind(model_name="llama-2-7b")
llama_70b = LlamaServer.options(ray_actor_options={"TPU-v4-32-head": 1}).bind(model_name="llama-2-70b")
gradio = MyGradioServer(LlamaServer.bind(model_name="llama-2-7b"))