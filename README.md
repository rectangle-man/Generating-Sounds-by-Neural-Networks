# Variational Autoencoder for Sound Regeneration

This repository contains a project for regenerating spoken digits from the Free Spoken Digit Dataset (FSDD) using a **Variational Autoencoder (VAE)**. The aim is to generate audio representations from spectrogram inputs, using the FSDD, a dataset of recorded spoken digits ('0' through '9') in different voices.

## Project Overview

In this project, we utilize a **Variational Autoencoder (VAE)** to encode audio spectrograms into a lower-dimensional latent space, then decode them back to the original spectrograms to regenerate the spoken digits. This model has two main components: 
1. **Encoder** - Compresses the input spectrogram into a latent representation.
2. **Decoder** - Reconstructs the spectrogram from the latent space, allowing for sound generation.

Once reconstructed, the spectrograms are converted back to waveforms, giving us regenerated audio that mimics the original spoken digits.

---

## Dataset: Free Spoken Digit Dataset (FSDD)

The Free Spoken Digit Dataset (FSDD) is a simple audio/speech dataset consisting of recordings of spoken digits in English, covering digits from '0' to '9'. Each digit is spoken multiple times by different speakers, providing variability in the dataset.

- **Total samples**: 3,000+
- **Sample rate**: 8kHz
- **File format**: `.wav`
- **Dataset link**: [FSDD GitHub Repository](https://github.com/Jakobovski/free-spoken-digit-dataset)

For this project, the dataset is preprocessed by converting the audio into **spectrograms**, which represent the frequency domain features of the audio, making it suitable for input to the VAE.

---

### Key Files:
- **autoencoder.py**: Contains the implementation of the Variational Autoencoder, with both the encoder and decoder architectures defined.
- **soundgenerator.py**: Implements functions to convert generated spectrograms back into audio format.
- **generator.py**: A script to test the trained VAE and regenerate sounds from the latent space.
- **preprocess.py**: Preprocessing script to convert the raw audio files from the FSDD into spectrograms that are fed into the VAE.
- **train.py**: Script to train the VAE on the preprocessed spectrogram data.

---

## Model Architecture

The **Variational Autoencoder (VAE)** used in this project follows the typical VAE structure with modifications suited for handling audio spectrogram data.

- **Encoder**: 
  - The encoder takes in a spectrogram as input and passes it through a series of convolutional layers. It reduces the dimensionality to a compact latent representation (mean and log-variance).
  
- **Latent Space**: 
  - The VAE introduces a probabilistic component to the latent space, ensuring the latent vectors follow a Gaussian distribution.
  
- **Decoder**:
  - The decoder takes in a sample from the latent space and reconstructs the original spectrogram, reversing the encoding process through deconvolutional layers.

- **Loss Function**: 
  - The loss is a combination of reconstruction loss (mean squared error between original and reconstructed spectrograms) and a KL-divergence loss (ensuring the latent space follows the desired Gaussian distribution).

---

## Preprocessing

The audio files are first transformed into spectrograms, as they represent the time-frequency domain of the audio signal. The steps include:
1. **Loading Audio**: Each `.wav` file from the FSDD is loaded and resampled to 8kHz.
2. **Generating Spectrograms**: We generate log-scaled Mel spectrograms using a Short-Time Fourier Transform (STFT).
3. **Normalization**: The spectrograms are normalized to a range that allows for efficient training of the VAE.

---

## Training the VAE

The model is trained to minimize both the reconstruction loss (to ensure that the generated spectrograms are as close as possible to the original ones) and the KL-divergence loss (to ensure the latent space follows a normal distribution).

- **Training Data**: Preprocessed FSDD spectrograms.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Epochs**: The model is trained for multiple epochs until the loss converges.

To train the model, use the `train.py` script
