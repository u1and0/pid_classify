# 品名と型式から品番カテゴリをAIで予測する
# usage:
#   docker run -it --rm -v `pwd`/data:/work/data -p 8880:8880 u1and0/pid_classify
#
# data以下にpidmaster.csvという名前の品番マスタ一覧のCSVファイルが必要です。

# ビルドコンテナ
FROM python:3.11.0-slim-bullseye as builder
WORKDIR /opt/app
RUN apt-get update &&\
    apt-get upgrade -y &&\
    apt-get install -y libfreetype6-dev \
                        libatlas-base-dev \
                        liblapack-dev
# COPY requirements.txt /opt/app  # for update image
COPY requirements.lock /opt/app
RUN pip install --upgrade -r requirements.lock

# 実行コンテナ
FROM python:3.11.0-slim-bullseye as runner
WORKDIR /work
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

RUN useradd -r classify_user
COPY main.py /work
COPY pid_classify.py /work
COPY templates/index.html /work/templates/index.html
COPY static/main.js /work/static/main.js
RUN chmod -R +x /work/main.py

USER classify_user
ENV PYTHONPATH="/work"
EXPOSE 8880
CMD ["python", "main.py"]

LABEL maintainer="u1and0 <e01.ando60@gmail.com>" \
      description="品名と型式から品番カテゴリをAIで予測する" \
      version="u1and0/pid_classify:v0.2.0"
