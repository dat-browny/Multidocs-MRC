# Use python as parent image
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04

LABEL maintainer="Le Thanh Tuan"

# Environment
ENV LANG=en_US.utf8
ENV LANG C.UTF-8

# Set Variable
ENV QUEUE_KEY="knowledge_grounded_r_g_chatbot_queue"
ENV OUTPUT_KEY="knowledge_grounded_r_g_chatbot_output"
ENV MODEL_PATH="mrc_models"


# Listen on port 5000
EXPOSE 5000

# Install basic packages and miscellaneous dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    wget \
    curl \
    bzip2 \
    vim \
    ffmpeg \
    unzip \
    alien \
    libaio1\
    libsm6 libxext6 libxrender-dev\
    git \
    gunicorn \
    default-jre \
    redis-server \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN apt update -y
# Create env
RUN conda clean --all --yes
RUN conda create -n vnpt python=3.7
RUN echo "source activate vnpt" > ~/.bashrc
ENV PATH /opt/conda/envs/vnpt/bin:$PATH

# Copy app source code to image 
RUN mkdir /multi_document_mrc \
    && mkdir /multi_document_mrc/multi_document_mrc \
    && mkdir $MODEL_PATH

COPY requirements.txt /multi_document_mrc/requirements.txt
RUN /bin/bash -c  "source activate vnpt && \
    cd /multi_document_mrc/ && pip install -r requirements.txt"

COPY multi_document_mrc /multi_document_mrc/multi_document_mrc

COPY setup.py /multi_document_mrc/setup.py
COPY run_service.sh /multi_document_mrc/run_service.sh
COPY convert_py_to_cy.py /multi_document_mrc/convert_py_to_cy.py

# Set workdir
WORKDIR /multi_document_mrc

RUN /bin/bash -c  "source activate vnpt && \
    python setup.py build_ext --inplace && \
    python convert_py_to_cy.py build_ext --inplace && \
    pip install -e ."


COPY $MODEL_PATH /multi_document_mrc/$MODEL_PATH

RUN chmod +x run_service.sh
CMD ["./run_service.sh"]

