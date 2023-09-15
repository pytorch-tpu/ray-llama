# Train
Examples for training LLaMa2 on Cloud TPUs with HuggingFace, Ray, and PyTorch/XLA with SPMD.

This folder consists of:
- [configs](configs/): the model definition of the various LLaMa-2 configurations, including a 2B variant for quick development work,
- [llama_hf.py](llama_hf.py), an adaptation of `run_clm.py` with PyTorch/XLA:SPMD support and simple refactoring for visual demonstration purposes, and
- [main.py](main.py), a full-fledged production job that can run on Ray Jobs.


These examples train from scratch and serve as a reference point for you to get started with Ray on TPUs.

To get started, spin up your training cluster:

```
$ ray up -y cluster/train.yaml
```

and once it's up and running, you can use

```
$ ./scripts/submit_train.sh
```

to submit the training job. Then you can run

```
$ ray dashboard cluster/train.yaml
```

and go to http://localhost:8265 to view the Job logs.
