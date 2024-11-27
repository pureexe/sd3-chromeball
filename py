#!/bin/bash

# enable proxy
ssh -D 1081 -N -f pakkaponp@ist-frontend-001


singularity shell --bind /ist:/ist --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ --env HF_HOME=/ist/users/$USER/.cache/huggingface --env HTTP_PROXY='socks5h://localhost:1081' --env HTTPS_PROXY='socks5h://localhost:1081' /ist/ist-share/vision/pakkapon/singularity/diffusers0310v4.sif
