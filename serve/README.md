# Serve
Examples for serving LLaMa2 on Cloud TPUs with Ray Serve.

Before running, make sure you [set up your Ray serve cluster](../cluster/):

```
ray up -y cluster/serve.yaml
```

This sample relies on https://github.com/facebookresearch/llama/tree/llama_v2 (but with a few Google Cloud/XLA improvements).


By default, this code will NOT load a checkpoint. Please ensure that you [request for access](https://github.com/facebookresearch/llama/tree/llama_v2#download) to the checkpoint (and go through the Meta AI License).

Once done, you can upload this to a GCS bucket and set this as your checkpoint path within `llama_serve.py`. This should help simplify the setup.

Note: This currently only supports serving the 7B model.

To deploy this model, run:

```
./scripts/start_gradio.sh
```

to submit the serve deployment, attach to the GradIO deployment via

```
$ ray attach -p 8000 cluster/serve.yaml
```

and go to http://localhost:8000 to view the GradIO deployment.