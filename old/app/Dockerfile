# Base image
FROM ubuntu:20.04

ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    python3-opencv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN apt-get update && apt-get install -y pip cmake

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt


# Set the working directory
WORKDIR /app

# Copy over the application code
COPY . .

# Start the application
CMD ["python3", "app.py"]