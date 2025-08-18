# Use Debian as the base image
FROM debian:latest

# Install dependencies (no sudo required)
RUN apt-get update && apt-get install -y \
    clang \
    gnuplot \
    libjson-c-dev \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    mkdir -p /app && \
    chown -R $USERNAME:$USERNAME /app

# Set working directory
WORKDIR /app

# Switch to the non-root user
USER $USERNAME

# Copy the C source file with correct ownership
COPY --chown=$USERNAME:$USERNAME neural_web64.c ./

# Copy the custom_embeddings.txt file with correct ownership
COPY --chown=$USERNAME:$USERNAME replace/with/your/own/path/custom_embeddings.txt ./

# Copy the vocabulary.txt file with correct ownership
COPY --chown=$USERNAME:$USERNAME replace/with/your/own/path/vocabulary.txt ./

COPY --chown=$USERNAME:$USERNAME /replace/with/path/to/you/have/to/also/change/the/c/code/to/include/it/definitions.h ./

# Compile the C program using Clang
RUN clang++ -std=c++17 -O2 -Wall -Wextra \
  -o neural_web64 neural_web.cpp \
  -I/usr/include \
  -ljson-c -lcurl -lm

# Set the default command to run the program
CMD ["./neural_web"]
