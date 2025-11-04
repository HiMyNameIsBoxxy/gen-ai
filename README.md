# Generative Models (FastAPI + Docker)

This project provides a FastAPI-based API that runs multiple deep learning models for both **text** and **image generation**, all containerized with Docker for easy deployment.

---

## 🧩 Available Models

### Bigram Language Model
- Generates word sequences based on bigram probabilities.
- Simple yet effective demonstration of probabilistic language modeling.
- Endpoint: `/generate`

### spaCy Embeddings
- Computes semantic embeddings of text using the **`en_core_web_md`** model.
- Useful for similarity search or NLP preprocessing tasks.
- Endpoint: `/embed`

### Convolutional Neural Network (CNN)
- A trained **CNN on CIFAR-10** predicts the class of uploaded images.
- Accepts `.jpg` / `.png` image files.
- Returns the predicted class label and confidence score.
- Endpoint: `/predict_cnn`

### Generative Adversarial Network (GAN)
- Trained on **MNIST** to generate realistic handwritten digits.
- Generates a grid of digits from random noise input.
- Endpoint: `/generate_gan`

### Diffusion Model
- Trained on **CIFAR-10-like** data to perform **denoising diffusion probabilistic modeling**.
- Generates high-quality, diverse images by progressively denoising random noise.
- Uses a learned noise scheduler and reverse diffusion sampling.
- Endpoint: `/generate_diffusion`

### Energy-Based Model (EBM)
- Trained on grayscale **CIFAR-10-like** images using an **energy function** that scores image realism.
- Uses **Langevin dynamics sampling** to iteratively generate new samples.
- Unlike GANs or VAEs, EBMs don’t have a direct generator; instead, they refine noise via energy gradients.
- Endpoint: `/generate_ebm`
---

## Requirements
- [Docker](https://docs.docker.com/get-docker/) installed on the host machine  
- Internet connection to pull base images  

---

## Build the Docker Image
Clone this repository and start your Docker.
```bash
git clone https://github.com/HiMyNameIsBoxxy/gen-ai.git
cd gen-ai
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
