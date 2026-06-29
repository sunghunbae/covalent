#!/usr/bin/bash

cd ..
docker run -it --rm --gpus all -v .:/home/${USER} aimnet2:nse bash
