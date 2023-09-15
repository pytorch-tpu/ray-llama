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


_MAX_BATCH_SIZE = 1

app = FastAPI()

RUNTIME = {
    "env_vars": {"PJRT_DEVICE": "TPU"}
}

# Replace this with a GCS path
_CHECKPOINT_PATH = ""


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
            f"[LLaMa7B]: {result}"
        )


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,
    },
    ray_actor_options={"TPU": 4})
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
        import torch
        import torch_xla
        import torch_xla.runtime as xr
        xr.use_spmd()

        import os
        import socket
        from llama import Llama

        self._hostname = socket.gethostname()
        self._model_name = model_name

        print(f"Using model: {model_name}.")
        self.generator = Llama.build(
            ckpt_dir=_CHECKPOINT_PATH,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            dynamo=dynamo)
        self._max_gen_len = max_gen_len
        self._temperature = temperature
        self._top_p = top_p

    def generate_batch(self, prompts: Iterable[str]) -> List[str]:
        import torch
        with torch.no_grad():
            results = self.generator.text_completion(
                prompts,
                max_gen_len=self._max_gen_len,
                temperature=self._temperature,
                top_p=self._top_p,
            )
            return results

    def __repr__(self):
        return f"[{self._model_name}-shard]: "


gradio = MyGradioServer(LlamaServer.bind(model_name="llama-2-7b"))