#!/usr/bin/bash

cd ..
docker run -it --rm --gpus all -v .:/home/${USER} orb_models:bitnami bash
