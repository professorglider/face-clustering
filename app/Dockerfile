FROM python:3.10

WORKDIR ./app

# Install dependencies
COPY requirements.txt .
RUN apt-get -o Debug::pkgProblemResolver=yes update && apt-get upgrade
RUN apt-get install libgtk2.0-dev pkg-config
RUN apt-get -y install cmake
RUN python3 -m pip install -r requirements.txt

# Copy source code
COPY . .

# Start command
CMD ["python", "face-recognition-service.py"]
