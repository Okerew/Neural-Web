#!/bin/bash

VERSION="1.0"

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or use sudo"
  exit
fi

OS="$(uname)"
echo "Detected OS: $OS"

HEADER="key_functions.h"

if [[ "$OS" == "Darwin" ]]; then
    # macOS: Cellar installation
    CELLAR="/usr/local/Cellar/neural_web/$VERSION"
    LIB_DIR="$CELLAR/lib"
    INCLUDE_DIR="$CELLAR/include/neural_web"
    mkdir -p "$LIB_DIR" "$INCLUDE_DIR"

    # Metal build
    METAL_LIB="libneural_web.dylib"

    cp "$METAL_LIB" "$LIB_DIR/"
    ln -sf "$LIB_DIR/$METAL_LIB" "/usr/local/lib/$METAL_LIB"

    # Copy header
    cp "$HEADER" "$INCLUDE_DIR/"

    # Copy README
    cp "README.md" "$CELLAR/"

elif [[ "$OS" == "Linux" ]]; then
    # Linux: C build always, CUDA build if available
    echo "Installing C and CUDA builds to /usr/local/bin and /usr/local/lib"

    # Include dir for Linux
    INCLUDE_DIR="/usr/local/include/neural_web"
    mkdir -p "$INCLUDE_DIR"
    cp "$HEADER" "$INCLUDE_DIR/"

    # C build
    C_LIB="libneural_web64.a"
    cp "$C_LIB" /usr/local/lib/

    # CUDA build: only if nvcc exists
    if command -v nvcc >/dev/null 2>&1; then
        echo "CUDA detected: installing CUDA build"
        CUDA_EXE="neural_web_cu"
        CUDA_LIB="libneural_web_cu.a"
        cp "$CUDA_LIB" /usr/local/lib/
    else
        echo "CUDA not found: skipping CUDA build installation"
    fi
fi

echo "Installation complete!"
