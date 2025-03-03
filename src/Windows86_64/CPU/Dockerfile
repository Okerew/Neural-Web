# Use Windows Server Core as the base image
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Set PowerShell as the default shell
SHELL ["powershell", "-Command"]

# Install Chocolatey
RUN Set-ExecutionPolicy Bypass -Scope Process -Force; \
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; \
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Clang, Make, and json-c
RUN choco install -y llvm make json-c

# Set working directory
WORKDIR C:\app

# Copy source code and dependencies
COPY neural_web64.c ./
COPY custom_embeddings.txt ./
COPY vocabulary.txt ./

# Compile the C program with json-c
RUN clang -o neural_web.exe neural_web64.c -I"C:\ProgramData\chocolatey\lib\json-c\include" -L"C:\ProgramData\chocolatey\lib\json-c\lib" -ljson-c -lm

# Set the default command to run the program
CMD ["C:\\app\\neural_web.exe"]
