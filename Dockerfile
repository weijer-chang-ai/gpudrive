FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
# Set non-interactive frontend to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PATH="/workspace/home/.local/bin:$PATH"

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    vim \
    emacs \
    build-essential \
    wget \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    mesa-common-dev \
    libc++1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
# Create conda environment with Python 3.11
RUN conda create -y -n gpudrive python=3.11 && \
    echo "source activate gpudrive" > ~/.bashrc
ENV PATH /opt/conda/envs/gpudrive/bin:$PATH
ENV CONDA_DEFAULT_ENV gpudrive
ENV CONDA_PREFIX /opt/conda/envs/gpudrive
# Setup working directory
WORKDIR /workspace
# Create mount point for local gpudrive repository
# The actual repository will be mounted here at runtime
RUN mkdir -p /workspace/gpudrive
# Set environment variables for GPUDrive
ENV PYTHONPATH=/workspace/gpudrive:${PYTHONPATH}
ENV MADRONA_MWGPU_KERNEL_CACHE=/workspace/gpudrive/gpudrive_cache
WORKDIR /workspace/gpudrive
# Create a directory for copied code
RUN mkdir -p /workspace/code
WORKDIR /workspace
ENV DEBIAN_FRONTEND=dialog
# Create an entrypoint script that builds and installs GPUDrive from the mounted source
RUN echo '#!/bin/bash\n\
    set -e\n\
    echo "Building GPUDrive from mounted source..."\n\
    cd /workspace/gpudrive\n\
    \n\
    # Create build directory if it doesn'\''t exist\n\
    mkdir -p build\n\
    cd build\n\
    \n\
    # Configure with cmake\n\
    cmake .. -DCMAKE_BUILD_TYPE=Release \\\n\
    -DPython_ROOT_DIR=/opt/conda/envs/gpudrive \\\n\
    -DPython_EXECUTABLE=/opt/conda/envs/gpudrive/bin/python \\\n\
    -DPython_INCLUDE_DIR=/opt/conda/envs/gpudrive/include/python3.11\n\
    \n\
    # Build\n\
    make -j$(nproc) madrona_gpudrive\n\
    cd ..\n\
    \n\
    # Install GPUDrive and dependencies\n\
    echo "Installing GPUDrive..."\n\
    /opt/conda/envs/gpudrive/bin/pip install --no-cache-dir -e .\n\
    /opt/conda/envs/gpudrive/bin/pip install --no-cache-dir pufferlib>=2.0.6 torch==2.6.0 nvidia-cuda-runtime-cu12==12.4.127 jupyter>=1.1.1 jupyterhub>=5.3.0 '\''numpy<2.0'\''\n\
    \n\
    # Install JAX and Wandb\n\
    echo "Installing JAX and Wandb..."\n\
    /opt/conda/envs/gpudrive/bin/pip install --no-cache-dir jaxlib==0.5.3 jax==0.5.3 '\''wandb[media]'\''\n\
    \n\
    echo "GPUDrive build and installation complete!"\n\
    \n\
    # Activate conda environment and start bash\n\
    source /opt/conda/bin/activate gpudrive\n\
    exec "$@"\n\
    ' > /workspace/entrypoint.sh && chmod +x /workspace/entrypoint.sh
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["/bin/bash"]