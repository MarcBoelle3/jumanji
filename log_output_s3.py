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
import glob

from s3fs.core import S3FileSystem


def main():
    # Initialiser S3
    s3 = S3FileSystem(client_kwargs={"endpoint_url": os.environ.get("S3_ENDPOINT")})
    output_path = os.environ.get("AICHOR_OUTPUT_PATH")

    # Chercher tous les fichiers .qdstrm dans /tmp/
    for file_path in glob.glob("/tmp/*.qdstrm"):
        filename = os.path.basename(file_path)
        s3_path = os.path.join(output_path, filename)

        print(f"Uploading {file_path} to {s3_path}")
        s3.put_file(file_path, s3_path)


if __name__ == "__main__":
    main()
