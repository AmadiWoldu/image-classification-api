# Image Classification API

A FastAPI-based Image Classification Service built using PyTorch and ResNet18. The API accepts image uploads, preprocesses them, and returns the predicted image class along with confidence scores.

## Features

* Image classification using ResNet18
* FastAPI REST API
* Image preprocessing pipeline
* Supports common image formats:

  * JPG
  * JPEG
  * PNG
  * BMP
  * GIF
  * WEBP
* Model training with PyTorch
* Experiment tracking with Weights & Biases (W&B)
* Robust error handling

---

## Project Structure

```text
image-classification-api/
│
├── api/
│   ├── app.py
│   └── schema.py
│
├── model/
│   ├── artifacts/
│   │   └── model_state_dict.pth
│   ├── inference.py
│   ├── preprocess.py
│   └── train.py
│
├── utils/
│   └── logger.py
│
├── data/
│
├── experiment/
│   └── experiment.ipynb
│
├── app.log
├── requirements.txt
└── README.md
```

---

## Dataset

The model is trained on the CIFAR-10 dataset.

Classes:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

---

## Model Architecture

* ResNet18 (Pretrained on ImageNet)
* Final classification layer modified for 10 CIFAR-10 classes
* Fine-tuning using PyTorch

---

## Installation

Create a virtual environment:

```bash
python -m venv image-classification-env
```

Activate environment:

Windows:

```bash
image-classification-env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run:

```bash
python model/train.py
```

The trained model will be saved to:

```text
model/artifacts/model_state_dict.pth
```

---

## Running the API

Start the FastAPI server:

```bash
uvicorn api.app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "running"
}
```

---

### Predict Image

```http
POST /predict
```

Accepts:

* JPG
* JPEG
* PNG
* BMP
* GIF
* WEBP

Response:

```json
{
  "prediction": "cat",
  "confidence": 98.74
}
```

---

## Example Usage

Using Swagger UI:

1. Open `/docs`
2. Select `POST /predict`
3. Upload an image
4. Execute request
5. View prediction result

---

## Technologies Used

* Python
* FastAPI
* PyTorch
* TorchVision
* PIL (Pillow)
* NumPy
* Weights & Biases
* Uvicorn

---

## Future Improvements

* Batch image prediction
* Top-K predictions
* Docker support
* Model versioning
* Cloud deployment (AWS/GCP)
* Support for custom datasets

---

## Author

Built as part of an ML Engineering and MLOps learning roadmap.
