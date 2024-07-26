# Baker Womack
# docker build -t datagen .
# docker run -p 30082:30082 --name datagen datagen

# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 30082 available to the world outside this container
EXPOSE 30082

# Run endpoint.py when the container launches
CMD ["python", "endpoint.py"]