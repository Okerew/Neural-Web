# Use Debian as the base image
FROM debian:latest

# Install dependencies (no sudo required)
RUN apt-get update && apt-get install -y \
    clang \
    gnuplot \
    libjson-c-dev \
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

# Compile the C program using Clang
RUN clang -o neural_web neural_web.c -I/usr/include -ljson-c -lm

# Set the default command to run the program
CMD ["./neural_web"]
