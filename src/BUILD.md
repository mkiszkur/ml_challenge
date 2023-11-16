
# image creation

## With poodman
podman build \
  -t challenge:latest \
  -f Containerfile \
  .

## With docker
docker build \
  -t challenge:latest \
  -f Containerfile \
  .
# run the container

## With podman
podman run -it \
  --name challenge \
  challenge:latest

## With docker
docker run -it \
  --name challenge \
  challenge:latest

# stop container
podman stop challenge

# start container
podman start challenge

# connect to the container via shell
podman exec -it challenge /bin/bash

