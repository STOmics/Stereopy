FROM python:3.10-bullseye

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /

RUN pip install pipx && \
    pipx install swe-rex && \
    pipx ensurepath

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    anndata \
    h5py \
    loompy \
    flake8 \
    pytest \
    shapely \
    colorcet \
    seaborn \
    matplotlib \
    scikit-learn

SHELL ["/bin/bash", "-c"]
ENV PATH="$PATH:/root/.local/bin/"
