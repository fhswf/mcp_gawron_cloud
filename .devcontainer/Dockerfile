FROM mcr.microsoft.com/devcontainers/python:dev-3.13-bookworm

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y curl \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* 
    
RUN curl -LsSf https://astral.sh/uv/install.sh | sh