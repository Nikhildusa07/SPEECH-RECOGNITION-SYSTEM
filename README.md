# SPEECH-TO-TEXT SYSTEM

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: DUSANIKHIL

*INTERN ID*: C0DF113

*DOMAIN*: SPEECH RECOGNITION / NATURAL LANGUAGE PROCESSING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH





#  Speech to Text System

SpeechToTextSystem is a Flask-based web application that converts spoken audio into accurate text using a hybrid approach combining **Wav2Vec2** (deep learning) and **Google SpeechRecognition** (cloud-based API). Users can upload or record audio directly in the browser and receive fast, reliable transcriptions.

---

##  Features

-  Upload or record speech using your browser
-  Hybrid speech recognition (Wav2Vec2 + Google SpeechRecognition)
-  View and copy accurate transcriptions
-  Fast response time with clean UI
-  Built with Flask, Hugging Face Transformers, and WebAudio API

---

##  How It Works

- **Wav2Vec2**: A transformer-based model trained on large audio datasets for deep learning speech recognition.
- **Google SpeechRecognition**: Cloud-based API used as a backup or enhancer for improved transcription.
- The app selects or merges results from both methods for better accuracy.

---

##  Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
