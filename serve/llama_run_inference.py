"""LLaMA Ray runner without Serve.

This should generally be easier to iterate on than a full serve deployment.

"""
from typing import Iterable, List
import time
import ray


_VALID_MODELS = [
    "llama-2-7b",
    "llama-2-13b",
    "llama-2-70b",
]
_MAX_BATCH_SIZE = 1
_CHECKPOINT_PATH = "gs://ray-llama-demo"
_ENABLE_VERBOSE_LOGGING = True
_LOAD_CHECKPOINT = True


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
        print(f"Initializing model: {model_name}.")
        self._host_name = socket.gethostname()
        self._model_name = model_name
        self._tokenizer_path = tokenizer_path
        self._max_batch_size = max_batch_size
        self._max_seq_len = max_seq_len
        self._max_gen_len = max_gen_len
        self._temperature = temperature
        self._top_p = top_p
        self._dynamo = dynamo
        if _LOAD_CHECKPOINT:
            self._ckpt_dir = os.path.join(_CHECKPOINT_PATH, model_name)
        else:
            self._ckpt_dir = ""
        self._worker_id = worker_id

    def __repr__(self) -> str:
        """Returns the actor logger prefix."""
        return f"LLaMAActor{self._model_name}::{self._host_name}"

    def initialize(self):
        """Initializes the LLaMA generator."""
        import os
        os.environ["PJRT_DEVICE"] = "TPU"
        if _ENABLE_VERBOSE_LOGGING:
            os.environ["TPU_STDERR_LOG_LEVEL"] = "0"
            os.environ["TPU_MIN_LOG_LEVEL"] = "0"
            os.environ["TF_CPP_VMODULE"] = "xla_graph_executor=5,pjrt_computation_client=3"
        import torch
        import torch_xla
        import torch_xla.runtime as xr
        from llama import Llama
        print("Building generator")
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

@ray.remote
class LlamaServer:
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
        self._model_name = model_name
        self._tokenizer_path = tokenizer_path
        self._max_batch_size = max_batch_size
        self._max_seq_len = max_seq_len
        self._max_gen_len = max_gen_len
        self._temperature = temperature
        self._top_p = top_p
        self._dynamo = dynamo

    def initialize(self):
        assert self._model_name in _VALID_MODELS
        tpu_pod_name = ray.util.accelerators.tpu.get_current_pod_name()
        num_tpu_pod_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
        assert ray.available_resources()[tpu_pod_name] == num_tpu_pod_hosts
        actor_def = LlamaTpuActor.options(resources={tpu_pod_name: 1, "TPU": 4})
        print("Creating TPU VM shards.")
        try:
            self._shards = [actor_def.remote(
                model_name=self._model_name,
                worker_id=i,
                tokenizer_path=self._tokenizer_path,
                max_batch_size=self._max_batch_size,
                max_seq_len=self._max_seq_len,
                max_gen_len=self._max_gen_len,
                temperature=self._temperature,
                top_p=self._top_p,
                dynamo=self._dynamo,
            ) for i in range(num_tpu_pod_hosts)]
            print("Created shards")
            print("Initializing shards")
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


def main():
    print("Initializing Ray")
    ray.init()

    print("Resources available to Ray: ", ray.available_resources())

    server = LlamaServer.options(
        resources={"TPU-v4-8-head": 1}).remote(
            model_name="llama-2-7b")
    try:
        print("Initializing the server.")
        start_time = time.time()
        ray.get(server.initialize.remote())
        print(f"Took {time.time() - start_time}s to initialize.")

        print("Running some requests now...")
        start_time = time.time()
        ray.get(server.generate_batch.remote(["I believe the meaning of life is"]))
        print(f"Took {time.time() - start_time}s to run.")
    except Exception as e:
        print("Captured failure: ", e)
        print("Shutting down the workload...")
        ray.shutdown()


if __name__ == "__main__":
    main()
