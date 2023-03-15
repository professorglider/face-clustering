# Base image
FROM ubuntu:22.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y ffmpeg libsm6 libxext6 software-properties-common cmake python3.10 pip libopencv-dev v4l-utils

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN setcap 'CAP_SYS_ADMIN+ep' $(which python3)

# Set the working directory
WORKDIR /app

# Copy over the application code
COPY . .

# Start the application
CMD ["python3", "app.py"]