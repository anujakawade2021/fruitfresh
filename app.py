import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("models/food_freshness.keras")
LABEL_PATH = Path("models/class_names.json")
INVENTORY_PATH = Path("food_inventory.csv")

st.set_page_config(page_title="Food Freshness Assistant", page_icon="🥗", layout="wide")
st.title("🥗 Food Freshness & Waste Reduction Assistant")
st.write("Estimate visible freshness patterns and manage food items to help reduce household waste.")
st.info("A photo cannot prove food safety. This model predicts visual categories only; never eat food solely because it is classified as fresh.")

if not MODEL_PATH.exists() or not LABEL_PATH.exists():
    st.error("Model files are missing. Follow README.md, add the dataset, then run: python train_model.py")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

TIPS = {
    "apple": "Store apples in the refrigerator for longer storage. Keep damaged fruit separate.",
    "banana": "Keep bananas in a suitable ventilated place. Refrigerate once ripe if appropriate.",
    "bitter_gourd": "Refrigerate in the vegetable compartment and use before it softens.",
    "capsicum": "Keep capsicum refrigerated and dry until use.",
    "orange": "Store oranges in a cool place or refrigerate for longer storage.",
    "tomato": "Keep unripe tomatoes at room temperature; refrigerate ripe tomatoes if needed.",
}

st.header("1. Check a food image")
uploaded_file = st.file_uploader("Upload a fruit or vegetable photo", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded food image", width=300)
        array = np.asarray(image.resize((160, 160)), dtype=np.float32)[None, ...]
        probabilities = model.predict(array, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        predicted_class = class_names[best_index]
        confidence = float(probabilities[best_index])

        parts = predicted_class.lower().split("_", 1)
        condition = parts[0] if len(parts) > 1 else "unknown"
        food_name = parts[1].replace("_", " ") if len(parts) > 1 else predicted_class

        st.subheader("Model prediction")
        st.write(f"Food category: **{food_name.title()}**")
        st.write(f"Predicted image class: **{predicted_class}**")
        st.metric("Model confidence", f"{confidence * 100:.1f}%")
        if confidence < 0.60:
            st.warning("The model is uncertain. Try a clear, well-lit photo of one food item.")
        elif condition == "fresh":
            st.success("The image resembles the dataset's fresh category.")
        elif condition == "stale":
            st.warning("The image resembles the dataset's stale category.")
        else:
            st.warning("The predicted category needs manual review.")

        st.caption("Confidence is the model's score for its selected class, not the probability that food is safe.")
        st.subheader("Waste reduction suggestion")
        st.write(TIPS.get(food_name, "Check storage guidance for this food and use older items first when appropriate."))
        st.write("Inspect food for unusual odor, mold, texture changes and other spoilage signs. Follow proper food-safety guidance.")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.divider()
st.header("2. Food inventory and use-first reminders")
if not INVENTORY_PATH.exists():
    pd.DataFrame(columns=["Food item", "Use-by date", "Notes"]).to_csv(INVENTORY_PATH, index=False)

inventory = pd.read_csv(INVENTORY_PATH)
with st.form("add_food"):
    food_item = st.text_input("Food item")
    use_by = st.date_input("Recorded expiry/use-by date", value=date.today())
    notes = st.text_input("Notes (optional)")
    submitted = st.form_submit_button("Add food item")
    if submitted:
        if not food_item.strip():
            st.error("Please enter a food item.")
        else:
            new_row = pd.DataFrame([{
                "Food item": food_item.strip(),
                "Use-by date": use_by.isoformat(),
                "Notes": notes.strip(),
            }])
            inventory = pd.concat([inventory, new_row], ignore_index=True)
            inventory.to_csv(INVENTORY_PATH, index=False)
            st.success("Food item added.")
            st.rerun()

if inventory.empty:
    st.write("Your inventory is empty. Add an item above.")
else:
    inventory["Use-by date"] = pd.to_datetime(inventory["Use-by date"], errors="coerce").dt.date
    def reminder(day):
        if pd.isna(day):
            return "Check date"
        remaining = (day - date.today()).days
        if remaining < 0:
            return "Date passed — check label and food safety"
        if remaining <= 3:
            return "Use soon — check condition and label"
        return "Plan usage"
    inventory["Reminder"] = inventory["Use-by date"].apply(reminder)
    st.dataframe(inventory, use_container_width=True)
    st.download_button(
        "Download inventory CSV",
        data=inventory.to_csv(index=False).encode("utf-8"),
        file_name="food_inventory.csv",
        mime="text/csv",
    )
    st.caption("Date reminders are based on the date you entered; they do not determine whether food is safe or spoiled.")

st.divider()
st.caption("AI/ML academic project • Predictions depend on dataset coverage and image quality.")
