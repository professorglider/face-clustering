FROM ubuntu:latest

WORKDIR ./web-view

# Install dependencies
COPY web-view/requirements.txt .
RUN apt-get -o Debug::pkgProblemResolver=yes update && apt-get upgrade
RUN apt-get -y install python3.10
RUN python3 -m pip install -r requirements.txt

# Copy source code
COPY . .

# Start command
CMD ["python", "app.py"]