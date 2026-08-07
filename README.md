Plant Disease Detection using Deep Learning and AWS

Farmers often struggle to identify plant diseases early, leading to reduced crop yield and economic losses. This project aims to automatically detect plant diseases from leaf images using a Convolutional Neural Network (CNN), allowing users to upload an image and receive an instant prediction.

Tech Stack:

Machine Learning
TensorFlow
Keras
NumPy
OpenCV
Pandas
Matplotlib

Backend: FastAPI

frontend: React

Deployment: 

Docker
Sagemaker AI
S3

DAtaset:

PlantVillage Dataset


Project Architecture:

                 User

                   │

                   ▼

            React Frontend

                   │

                   ▼

        FastAPI (optional)

                   │

          InvokeEndpoint API

                   │

                   ▼

     Amazon SageMaker Endpoint

                   │

                   ▼

      TensorFlow EfficientNetB0

                   │

                   ▼

        Disease + Confidence