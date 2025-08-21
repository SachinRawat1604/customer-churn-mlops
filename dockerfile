# use an official Python runtime as a parent image
FROM python:3.11-slim

#set working directory
WORKDIR /app

# Install system dependencies required for building Python packages.
# The 'build-essential' package provides the C compiler (cc) and other tools.
# The 'libzmq3-dev' package is needed for the pyzmq library.
# We also clean up the apt cache to keep the image size small.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libzmq3-dev && \
    rm -rf /var/lib/apt/lists/*

# copy the requirements.txt file into the container
COPY requirements.txt .

# install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy the application source code and the model into the container
COPY . .

# expose the port 8000 to all communications with the app 
EXPOSE 8000

# command to run the application when the container launches
# we use 0.0.0.0 to make the server accessible from outside the container
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

