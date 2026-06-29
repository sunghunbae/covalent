#!/usr/bin/bash

docker build \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg USER=$(whoami) \
  -t orb_models:bitnami \
  -f Dockerfile.bitnami .
