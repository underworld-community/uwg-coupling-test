FROM ghcr.io/julesghub/underworld2:2.17
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
RUN git clone --single-branch --branch main https://github.com/underworld-community/uwg-coupling-test.git \
 && chmod -R g+rwX uwg-coupling-test

