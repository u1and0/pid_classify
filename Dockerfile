# 品名と型式から品番カテゴリをAIで予測する
# usage:
#   docker run -it --rm -v `pwd`/data:/work/data -p 8880:8880 u1and0/pid_classify
#
# data以下にpidmaster.csvという名前の品番マスタ一覧のCSVファイルが必要です。

# Python build container
FROM python:3.13.5-slim-bullseye AS builder
RUN apt-get update &&\
    apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends\
                    libfreetype6-dev\
                    libatlas-base-dev\
                    liblapack-dev\
                    curl

# uvのインストール
# By default, uv is installed to ~/.local/bin
# https://docs.astral.sh/uv/reference/installer/
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# pyproject.tomlから最新のパッケージをインストール
# uv sync は pyproject.toml (およびロックファイル) に基づいて依存関係を同期します。
# Docker環境では、仮想環境を作成せずにシステムサイトパッケージに直接インストールするため
# --system フラグを使用します。
WORKDIR /app
COPY pyproject.toml /app/
RUN uv pip compile pyproject.toml --output-file requirements.lock &&\
    uv pip sync --system requirements.lock

# TypeScript build container
FROM node:20.19.3-bullseye-slim AS tsbuilder
COPY ./static /tmp/static
WORKDIR /tmp/static
RUN npm install -D typescript ts-node ts-node-dev
RUN npx tsc # || exit 0  # Ignore TypeScript build error

# 実行コンテナ
FROM python:3.13.5-slim-bullseye AS runner
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=tsbuilder /tmp/static/main.js /app/static/main.js

RUN useradd -r classify_user
COPY main.py /app/main.py
COPY pid_classify /app/pid_classify/
COPY templates/ /app/templates/
COPY static/favicon.png /app/static/favicon.png
RUN chmod -R +x /app/main.py

USER classify_user
ENV PYTHONPATH="/app"
EXPOSE 8880
CMD ["python", "main.py"]

LABEL maintainer="u1and0 <e01.ando60@gmail.com>" \
      description="品名と型式から品番カテゴリや諸口品番を予測する" \
      name="u1and0/pid_classify"
