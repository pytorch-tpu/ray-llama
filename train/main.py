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
import ray
from typing import Any, Mapping


RUNTIME = {
  "env_vars": {
    "PJRT_DEVICE": "TPU",
    "XLA_USE_SPMD": "1",
    "LIBTPU_INIT_ARGS": "--xla_tpu_spmd_rng_bit_generator_unsafe=1",
    "XLA_USE_BF16": "1",
    "XLA_IR_DEBUG": "1",
    "XLA_HLO_DEBUG": "1",
  }
}

ARGS_BY_CONFIG = {
    "7B": {
        # Expects to run on a v4-32
        "model_name": "llama2/7B",
        "per_device_train_batch_size": 16,
        "spmd_2d_sharding": 1,
    },
    "13B": {
        # Expects to run on a v4-64
        "model_name": "llama2/13B",
        "per_device_train_batch_size": 8,
        "spmd_2d_sharding": 1,
    },
    "70B": {
        # Expects to run on a v4-256
        "model_name": "llama2/70B",
        "per_device_train_batch_size": 16,
        "spmd_2d_sharding": 4,
        "num_train_epochs": 50,
    },
}


@ray.remote(resources={"TPU": 4})
def train_llama(hf_args: Mapping[str, Any]):
    import socket 
    import torch_xla
    import torch_xla.runtime as xr
    xr.use_spmd()

    from llama_hf import init_hf, train

    print(f"Initializing shard: {socket.gethostname()}")

    hf = init_hf(**hf_args)
    print("Starting to train")
    train(
        training_args=hf["training_args"],
        data_args=hf["data_args"],
        trainer=hf["trainer"],
        train_dataset=hf["train_dataset"])


def main():
    ray.init(runtime_env=RUNTIME)

    args = ARGS_BY_CONFIG["70B"]
    print("Args: ", args)

    num_available_tpus = int(ray.available_resources()["TPU"])
    num_hosts = num_available_tpus // 4
    print("Num hosts: ", num_hosts)

    try:
      print("Creating trainer shards.")
      train_handles = [train_llama.remote(args) for _ in range(num_hosts)]
      print("Started training.")
      training_results = ray.get(train_handles)
      print("Training results: ", training_results)
    except Exception as e:
      print(f"Caught failure: {e}")
      print("Shutting down")
      ray.shutdown()


if __name__ == "__main__":
    main()