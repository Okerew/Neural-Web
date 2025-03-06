# Use the NVIDIA CUDA base image with Debian
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install dependencies (no sudo required in Docker)
RUN apt-get update && apt-get install -y \
    gnuplot \
    libjson-c-dev \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user (optional, but avoids root execution)
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    chown -R $USERNAME:$USERNAME /app

# Set working directory
WORKDIR /app

# Switch to the non-root user
USER $USERNAME

# Copy the CUDA source file
COPY --chown=$USERNAME:$USERNAME neural_web.cu ./

# Copy the custom_embeddings.txt file with correct ownership
COPY --chown=$USERNAME:$USERNAME custom_embeddings.txt ./

# Copy the vocabulary.txt file with correct ownership
COPY --chown=$USERNAME:$USERNAME vocabulary.txt ./

# Compile the CUDA program using NVCC
RUN nvcc -o neural_web neural_web.cu -I/usr/include -ljson-c -lcurl

# Set the default command to run the program
CMD ["./neural_web"]
