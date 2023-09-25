# ray-llama
Examples for running Llama 2 on Ray with [Google Cloud TPUs](https://cloud.google.com/tpu).


## Folder structure
- [cluster](cluster/): sample YAML files for creating your Ray cluster.
- [notebooks](notebooks/): sample notebook to demonstrate interacting with Ray TPU clusters.
- [docker](docker/): sample Docker files for quick env setup.
- [serve](serve/): sample code for RayServe deployments.
- [train](train/): sample code for pretraining from scratch.
- [scripts](scripts/): sample scripts to automate common tasks.

## How to Get Started

To get started with this repo, a great option to start is to started with an interactive notebook environment. See [notebooks](notebooks/).

If you are interested in large scale training runs, see [train](train/) to get started.

If you are interested in serving, see [serve](serve/) to get started.


## Setting up Your Environment
To quickly set up your environment, you can run

```
$ ./scripts/set_project_info.sh
```

and supply a base GCR/Docker path and GCP project ID. This
will automatically set these values in cluster YAML files and scripts.