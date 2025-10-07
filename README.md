## Pneumonia Detection from Chest X‑rays (CNN + Streamlit)

An end‑to‑end project to detect Pneumonia from chest X‑rays using a Convolutional Neural Network (CNN) and a simple Streamlit web app for interactive inference.

### Demo

Sample X‑ray images used in this project:

Normal example:

<img width="1222" height="1121" alt="image" src="https://github.com/user-attachments/assets/ad17a749-ff53-498d-a88d-b44a9f27cd44" />


Pneumonia example (replace the filename below with any file from `test/PNEUMONIA/` on your machine if it doesn't render in your clone):

<img width="1090" height="1106" alt="image" src="https://github.com/user-attachments/assets/f5fa75f3-54f4-429c-af5a-e7d3d53efbf1" />



### Features
- CNN trained on 128×128 grayscale X‑ray images
- Binary classification: NORMAL vs PNEUMONIA
- Streamlit app for drag‑and‑drop image predictions
- Robust preprocessing that automatically matches the model’s expected input shape


## Project Structure

```
Xray/
  app.py                     # Streamlit app for inference
  CNN_Model.ipynb            # Notebook: data prep, training, evaluation
  pneumonia_cnn_model.h5     # Trained model (legacy H5 format)
  requirements.txt           # Python dependencies
  test/                      # Sample dataset (NORMAL / PNEUMONIA images)
```


## Getting Started

### 1) Environment

Install Python 3.10–3.12. Then install dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- On Windows + Python 3.12, TensorFlow >= 2.16 is required (the `requirements.txt` is already compatible).


### 2) Run the Web App

From the project root:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

Upload any chest X‑ray image (`.jpg`, `.jpeg`, `.png`) and the app will report:
- Predicted label: `NORMAL` or `PNEUMONIA`
- Confidence score


## Model Details

The model was trained in `CNN_Model.ipynb`. Key points:

- Input: 128×128×1 (grayscale)
- Architecture (Sequential):
  - Conv2D(32) → MaxPool
  - Conv2D(64) → MaxPool
  - Conv2D(128) → MaxPool
  - Flatten → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid)
- Loss: Binary Cross‑Entropy
- Optimizer: Adam
- Metrics: Accuracy
- Training time: 10 epochs on the dataset (train/val/test splits referenced in the notebook)

In the notebook, the model achieved around 86% test accuracy (see the final evaluation cells). The training curves and a confusion matrix are also produced in the notebook for deeper analysis.


## Inference Preprocessing

The app automatically adapts to the model’s expected input shape via `model.input_shape`:
- If the model expects 1 channel, the uploaded image is converted to grayscale and reshaped to `(1, H, W, 1)`
- If it expects 3 channels, the image stays in RGB and is reshaped to `(1, H, W, 3)`
- Images are resized to the model’s input size and normalized to `[0, 1]`


## Re‑training or Updating the Model

Open `CNN_Model.ipynb` and run cells end‑to‑end. To save the trained model:

```python
model.save('pneumonia_cnn_model.h5')  # legacy format used by this repo
# or the recommended Keras format
model.save('pneumonia_cnn_model.keras')
```

The app will prefer `pneumonia_cnn_model.keras` if present, and fall back to `pneumonia_cnn_model.h5`.


## Troubleshooting

- TensorFlow version errors on install
  - Ensure Python 3.10–3.12, then `pip install -r requirements.txt`.
- Model file not found
  - Place `pneumonia_cnn_model.h5` or `pneumonia_cnn_model.keras` in the project root (same folder as `app.py`).
- Input shape/channel mismatch
  - The app now auto‑handles channels. If you train a new model with different input size, the app will resize to the new `(H, W)` automatically.


## License

This project is provided for educational and research purposes. Ensure clinical decisions are made by qualified professionals and with appropriate regulatory approvals.


