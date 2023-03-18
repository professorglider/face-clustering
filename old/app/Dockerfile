# Base image
FROM python:3.10-slim-buster

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y ffmpeg libsm6 libxext6 software-properties-common cmake pip libopencv-dev v4l-utils

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Set the working directory
WORKDIR /app

# Copy over the application code
COPY . .

# Start the application
CMD ["python3", "app.py"]