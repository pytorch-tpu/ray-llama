# Cluster templates

This folder contains sample YAML files to launch Ray clusters for dev, serve and train.

Please ensure that you've already built your Docker image and pushed it to GCR (see [docker](../docker) instructions)
and specify the worker images and project IDs within these yaml files.

See [the Ray Cluster docs](https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-cli.html) for more information.


Sample command:
```
ray up -y dev.yaml
```

