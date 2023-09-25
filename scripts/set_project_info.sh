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

# Prompt inputs
read -p "Enter a base GCR image path: " gcr_image_path
read -p "Enter the GCP project string: " gcp_project
echo ""

# Validation (optional)
if [[ -z "$gcr_image_path" || -z "$gcp_project" ]]; then
  echo "All fields are required."
  exit 1
fi

echo "Processing scripts/build_docker.sh"
sed -i "s|GCR_PATH= |GCR_PATH=$gcr_image_path|g" scripts/build_docker.sh

echo "Processing cluster/dev.yaml"
sed -i "s|image: null|image: $gcr_image_path:train|g" cluster/dev.yaml
sed -i "s|project_id: null|project_id: $gcp_project|g" cluster/dev.yaml

echo "Processing cluster/train.yaml"
sed -i "s|image: null|image: $gcr_image_path:train|g" cluster/train.yaml
sed -i "s|project_id: null|project_id: $gcp_project|g" cluster/train.yaml

echo "Processing cluster/serve.yaml"
sed -i "s|image: null|image: $gcr_image_path:serve|g" cluster/serve.yaml
sed -i "s|project_id: null|project_id: $gcp_project|g" cluster/serve.yaml

echo "Configurations have been set!"