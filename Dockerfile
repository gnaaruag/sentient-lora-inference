
# Use a CUDA-enabled base image with Miniconda
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables to avoid the interactive timezone prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install Miniconda manually
ENV CONDA_DIR=/opt/conda
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    $CONDA_DIR/bin/conda clean --all --yes
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y libgl1-mesa-glx


# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install system packages (including tzdata) without prompting for timezone
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tzdata \
    git && \
    rm -rf /var/lib/apt/lists/*

# Set timezone to India (Asia/Kolkata)
RUN ln -fs /usr/share/zoneinfo/Asia/Kolkata /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN conda config --set channel_priority flexible

RUN conda env create -f environment.yaml


# Ensure CUDA and cuDNN are available in the container
RUN conda install -c conda-forge cudatoolkit=11.7 cudnn=8.2

# Set the shell to ensure environments are activated properly
SHELL ["conda", "run", "-n", "sentient-training", "/bin/bash", "-c"]

# Install additional dependencies

# Set the entrypoint to activate the Conda environment and run the script
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "sentient-training", "python", "-u", "script.py"]
