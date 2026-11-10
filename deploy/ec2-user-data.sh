#!/bin/bash
# User data do EC2 — cole este arquivo em "Advanced details > User data" ao
# criar a instância (AMI Ubuntu Server 24.04 LTS). Roda UMA vez, como root, na
# primeira inicialização. Log: /var/log/cloud-init-output.log
#
# Deixa a máquina pronta para `docker compose -f compose.serve.yml up`:
# swap, Docker Engine (+ plugins compose e buildx) e o repositório clonado.
set -euxo pipefail

# --- swap de 2 GB ---------------------------------------------------------
# MLflow + Streamlit + o build da imagem passam de 1 GB de RAM: numa t3.micro
# sem swap o build morre por OOM no meio do `uv sync`.
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# --- Docker ---------------------------------------------------------------
# O script oficial instala docker-ce já com os plugins compose e buildx
# (o `docker compose build` atual exige o buildx).
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker
usermod -aG docker ubuntu   # `docker` sem sudo a partir do próximo login

# --- código ---------------------------------------------------------------
if [ ! -d /home/ubuntu/MLOps ]; then
  sudo -u ubuntu git clone https://github.com/leoarjo/MLOps.git /home/ubuntu/MLOps
fi
sudo -u ubuntu mkdir -p /home/ubuntu/MLOps/artifacts
