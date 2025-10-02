# Bigram Text Generation API (FastAPI + Docker)


This project provides a FastAPI-based API that runs a **Bigram Language Model** & **spaCy**.  

**Bigram Language Model**
- Generates word sequences given a starting word and a desired length.

**spaCy**
- Adds functionality to compute semantic embeddings of text using the en_core_web_md model


---

## Requirements
- [Docker](https://docs.docker.com/get-docker/) installed on the host machine  
- Internet connection to pull base images  

---

## Build the Docker Image
Clone this repository and start your Docker.
```bash
git clone https://github.com/HiMyNameIsBoxxy/bigram_model.git
cd bigram_model
```

From the root project directory, build the docker image:

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
