FROM mcr.microsoft.com/devcontainers/python:dev-3.12-bookworm

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y curl \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && git lfs install \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* 

USER vscode    
RUN curl -LsSf https://astral.sh/uv/install.sh | sh  