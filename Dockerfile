#export Neptune credentials (secret variable in AIchor)
FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04@sha256:1c3cefb97f774264b9709eb209aa3a910ce5ba56889aa85d19d525a29ac01523

RUN apt-get update && apt-get install -y python3-pip

COPY requirements/ requirements/

RUN pip3 install --upgrade pip && \
    for req in requirements/*.txt; do pip3 install -r "$req"; done

COPY . .

# RUN --mount=type=secret,id=_env,dst=/etc/secrets/.env . /etc/secrets/.env \
#         && export NEPTUNE_API_TOKEN==$NEPTUNE_API_TOKEN
