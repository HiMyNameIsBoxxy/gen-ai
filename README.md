# Bigram Text Generation API (FastAPI + Docker)

This project provides a FastAPI-based API that runs a **Bigram Language Model**.  
The API can generate text sequences given a starting word and desired length.  
It is containerized with Docker so it can be run on any machine without extra setup.

---

## Requirements
- [Docker](https://docs.docker.com/get-docker/) installed on the host machine  
- Internet connection to pull base images  

---

## Build the Docker Image
Clone this repository and from the root project directory run:

```bash
docker build -t gen-ai .
```


## Run the Container
After building the image, start the API with:

```bash
docker run -p 8000:8000 gen-ai
```

## Test Link
Once the container is running, open your browser and go to:

```bash
http://127.0.0.1:8000/docs
```
