# Dockerfiles

This folder contains sample Dockerfiles for you to use that contains working environments for Llama train ([Dockerfile.train](Dockerfile.train)) and serve ([Dockerfile.serve](Dockerfile.serve)).

For your convenience, check out [scripts/](../scripts) for sample scripts to build and deploy these images to GCR:

```
./scripts/build.sh all
```

For serving - make sure that you request access to the tokenizer amd model checkpoints from https://github.com/facebookresearch/llama/tree/llama_v2, and copy the `tokenizer.model` file to this repository.
