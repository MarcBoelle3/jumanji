#!/usr/bin/env python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

import os
import sys
import warnings
from pathlib import Path

from hydra import compose, initialize
from s3fs.core import S3FileSystem

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from jumanji.training.train import train

warnings.filterwarnings("ignore")


def main() -> None:
    """Main function to run the training script."""
    # Set up environment and agent
    env = "job_shop_improvement"
    agent = "random"

    # Initialize Hydra and run training
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                f"env={env}",
                f"agent={agent}",
                "logger.type=terminal",
                "logger.save_checkpoint=true",
            ],
        )
        # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False):
        # with jax.profiler.trace("/tmp/jax-trace"):
        train(cfg)


if __name__ == "__main__":
    main()

    # for Nsight profiling
    s3 = S3FileSystem(client_kwargs={"endpoint_url": os.environ.get("S3_ENDPOINT")})
    s3.put_file(
        "/tmp/report.qdstrm", os.path.join(os.environ.get("AICHOR_OUTPUT_PATH"), "report.qdstrm")
    )
