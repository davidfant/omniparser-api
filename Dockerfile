FROM registry.hf.space/microsoft-omniparser:latest

USER root

RUN chmod 1777 /tmp \
    && apt update -q && apt install -y ca-certificates wget \
    && wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb && apt update -q \
    && apt install -y --no-install-recommends libcudnn8 libcublas-12-2

COPY app.py app.py
RUN python app.py
CMD ["python", "app.py"]