# Helper scripts

This folder contains helper scripts for common actions.

## Building the docker image(s) and deploying to GCR

Within `build_docker.sh`, make sure you replace line 3 which sets the `GCR_PATH` variable.

Let's say you set your `GCR_PATH=gcr.io/my-project/my-image`.

```
$ ./scripts/build_docker.sh all
```

will build and deploy these images to `gcr.io/my-project/my-image:train` and `gcr.io/my-project/my-image:serve`, which
you would then need to put within your [cluster/train.yaml](../cluster/train.yaml) and [cluster/serve.yaml](../cluster/serve.yaml)
files.


## Connecting to the dev cluster in Client mode

```
$ ./scripts/get_cluster_ip.sh

External IP address of the head node: 35.186.77.141
Internal IP address of the head node: 10.130.0.107


If developing from a notebook, connect to the Ray cluster as follows:

import ray

ray.init("ray://10.130.0.107:10001")
```

## Submitting the Training Job

Run

```
$ ./scripts/submit_train.sh

```

to submit the main training job, attach to the dashboard via

```
$ ray dashboard cluster/train.yaml
```

and go to http://localhost:8265 to view the Job logs.


## Submitting the Serving job

Run

```
$ ./scripts/start_gradio.sh
```

to submit the serve deployment, attach to the GradIO deployment via

```
$ ray attach -p 8000 cluster/serve.yaml
```

and go to http://localhost:8000 to view the GradIO deployment.
