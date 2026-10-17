# Food Freshness & Waste Reduction Assistant

Beginner-friendly AI/ML academic project using Python, TensorFlow/Keras, MobileNetV2 and Streamlit.

## Important dataset note
The dataset images are **not bundled** in this ZIP because the dataset is hosted by a third party and may have its own license/terms. Download it from:
https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables

Extract the downloaded archive. Place the 12 class folders directly inside `dataset/`, so the structure looks like:
```
dataset/
  fresh_apple/
  fresh_banana/
  fresh_bitter_gourd/
  fresh_capsicum/
  fresh_orange/
  fresh_tomato/
  stale_apple/
  stale_banana/
  stale_bitter_gourd/
  stale_capsicum/
  stale_orange/
  stale_tomato/
```
Exact folder names in the downloaded archive may differ; preserve the original class folder names, and ensure each folder contains images.

## Setup on Windows (VS Code terminal)

1. Install Python 3.11 if needed.
2. Open this folder in VS Code, then Terminal > New Terminal.
3. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Train the model:
   ```powershell
   python train_model.py
   ```
   The first run downloads pretrained MobileNetV2 weights, so internet is required.
6. Launch the app:
   ```powershell
   streamlit run app.py
   ```
7. Open the local URL printed by Streamlit, usually http://localhost:8501.

## What is included
- `train_model.py`: image dataset loading, transfer-learning model training, validation metrics, classification report and confusion matrix.
- `app.py`: image upload/prediction, storage suggestions, food inventory and date-based reminders.
- `requirements.txt`: Python package list.
- `dataset/`: empty placeholder folder; add the downloaded dataset here.
- `models/`: trained model and evaluation files will be created here.

## Evaluation
The script uses a seeded 80/20 train/validation split. Its printed metrics are validation results, not a separate independent test-set score. Do not claim a particular accuracy until you run the script and record your actual output. For stronger academic evaluation, create a separate test set before tuning the model.

## Limitations
- The classifier recognizes visual patterns in its training classes. It may misclassify unfamiliar food, lighting, or camera angles.
- The model's confidence score is not the probability that food is safe.
- Date reminders are based only on dates entered by the user.
- Do not eat food solely based on the model's output; follow food safety guidance and inspect the food appropriately.
