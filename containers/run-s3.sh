#!/usr/bin/bash

#docker run -d -it --rm --gpus all -v .:/home/${USER} aimnet2:nse python cli-run.py intact_ens10_s3.db --model isayevlab/aimnet2-nse
#docker run -d -it --rm --gpus all -v .:/home/${USER} aimnet2:nse python cli-run.py intact_ens10_s3_neutral.db

docker run -d -it --rm --gpus '"device=0"' -v .:/home/${USER} orb_models:bitnami python cli-run-orbmol.py intact_ens10_s3_neutral.db
