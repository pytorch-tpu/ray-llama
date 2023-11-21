#!/bin/bash
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
GCR_PATH=gcr.io/mlperf-high-priority-project/ray-llama2# fill this part out

if [ -z $GCR_PATH ]
then
  echo "build_docker.sh: please make sure to set the GCR_PATH variable."
fi

build_and_deploy() {
  GCR_DEST=$GCR_PATH:$1
  command="docker build . -f Dockerfile.$1 --network=host -t $GCR_DEST"
  echo "Building $GCR_DEST..."
  echo "Command: $command"
  $command

  echo "Built! Now pushing to GCR."
  docker push $GCR_DEST
}

# Function to display help message
show_help() {
  echo "Helper script to build and deploy Docker images."
  echo "Script usage: ./build.sh -train|serve|all."
}

# Parse command line arguments
while getopts ":h" opt; do
  case ${opt} in
    h )
      show_help
      exit 0
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      show_help
      exit 1
      ;;
  esac
done

# Shift to get the remaining arguments
shift $((OPTIND -1))

# Check remaining arguments
if [ "$#" -ne 1 ]; then
  echo "Exactly one argument required: cpu, gpu, or all."
  show_help
  exit 1
fi

# Main logic for building
case $1 in
  cpu)
    echo "Building for train..."
    build_and_deploy train
    ;;
  gpu)
    echo "Building for serve..."
    build_and_deploy serve
    ;;
  all)
    echo "Building for train and serve..."
    build_and_deploy train
    build_and_deploy serve
    ;;
  *)
    echo "Invalid argument: $1"
    show_help
    exit 1
    ;;
esac
