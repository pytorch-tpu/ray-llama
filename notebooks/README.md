# Notebooks

To get started with developing Ray on TPUs, you can spin up a development cluster that contains a single host TPU VM:

```
$ ray up -y cluster/dev.yaml
```

Once this is set up, you can easily find the IP binding via:
```
$ ./scripts/get_cluster_ip.sh
```