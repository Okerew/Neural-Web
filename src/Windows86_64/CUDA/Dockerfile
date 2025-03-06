# Use NVIDIA's CUDA base image for Windows
FROM nvcr.io/nvidia/cuda:12.2.2-devel-ps

# Set PowerShell as the default shell
SHELL ["powershell", "-Command"]

# Install Chocolatey (for package management)
RUN Set-ExecutionPolicy Bypass -Scope Process -Force; \
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; \
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install gnuplot and json-c
RUN choco install -y gnuplot json-c curl

# Set working directory
WORKDIR C:\app

# Copy source files
COPY neural_web.cu ./
COPY custom_embeddings.txt ./
COPY vocabulary.txt ./

# Compile the CUDA program using NVCC
RUN nvcc -o neural_web.exe neural_web.cu -lcurl -I"C:\ProgramData\chocolatey\lib\json-c\include" -L"C:\ProgramData\chocolatey\lib\json-c\lib" -ljson-c

# Set the default command to run the program
CMD ["C:\\app\\neural_web.exe"]
