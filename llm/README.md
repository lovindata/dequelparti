# Contribution

## Installation

Please install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```powershell
Start-Process 'Docker Desktop Installer.exe' -Wait -ArgumentList 'install', '--accept-license', '--installation-dir=D:\Application\Docker\installation-dir', '--hyper-v-default-data-root=D:\Application\Docker\hyper-v-default-data-root', '--windows-containers-default-data-root=D:\Application\Docker\windows-containers-default-data-root', '--wsl-default-data-root=D:\Application\Docker\wsl-default-data-root'
```

To verify NVIDIA GPUs with WSL2 usage.

```bash
docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

Please install VSCode and its extension(s):

- Docker

## Local build

To build local docker image:

```bash
docker build -t lovindata/private:dequelparti-llm-local -f "./Dockerfile" .
```

To investigate the content of the docker image:

```bash
docker run --rm -it --gpus=all --entrypoint /bin/sh lovindata/private:dequelparti-llm-local
```

To launch the docker image:

```bash
docker run --rm -it --gpus=all lovindata/private:dequelparti-llm-local
```

To clean docker build cache:

```bash
docker builder prune -af
```

To clean docker build cache & dandling images:

```bash
docker system prune -f
```

## Remote multi-platform build

To list, create, use and delete the multi-platform builder:

```bash
docker buildx ls
```

```bash
docker buildx create --use --name multi-platform-builder
```

```bash
docker buildx use multi-platform-builder
```

```bash
docker buildx rm multi-platform-builder
```

To build and push multi-platform docker images:

```bash
docker buildx build -t lovindata/private:dequelparti-llm-0.0.0 -t lovindata/private:dequelparti-llm-latest --platform linux/amd64,linux/arm64 --push -f "./Dockerfile" .
```

To clean docker buildx cache:

```bash
docker buildx prune -af
```
