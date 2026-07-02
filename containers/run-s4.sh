#!/usr/bin/bash

#docker run -d -it --rm --gpus all -v .:/home/${USER} aimnet2:nse python cli-run.py intact_ens10_s4.db --model isayevlab/aimnet2-nse
#docker run -d -it --rm --gpus all -v .:/home/${USER} aimnet2:nse python cli-run.py intact_ens10_s4.db

#docker run -d -it --rm --gpus all -v .:/home/${USER} aimnet2:nse python cli-run.py pruned_ens10_s4.db
#docker run -d -it --rm --gpus all -v .:/home/${USER} aimnet2:nse python cli-run.py pruned_C_ens10_s4.db

docker run -d -it --rm --gpus '"device=1"' -v .:/home/${USER} orb_models:bitnami python cli-run-orbmol.py intact_ens10_s4.db
