# 🐱🐶 Cat vs Dog Image Classifier – Local App with Streamlit

This is a simple image classification project that identifies whether an uploaded image is a **cat or dog** using a Convolutional Neural Network (CNN) with **MobileNetV2** (transfer learning). The project also includes a local **Streamlit-based app** that runs on your system and provides real-time predictions.

---

## 📁 Dataset

- **Source**: [Kaggle - Cat vs Dog Dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)
- Two folders: `Cat/` and `Dog/`

---

## 🔄 Project Workflow

1. ✅ **Data Preprocessing**:
   - Removed corrupted files
   - Resized all images to 224x224
   - Normalized pixel values (0–1)
   - Applied augmentation (rotation, flip, brightness, zoom)

2. ✅ **Model Building**:
   - Pre-trained MobileNetV2 as base
   - Added custom dense layers
   - Trained using binary crossentropy

3. ✅ **Model Evaluation**:
   - Used classification report, confusion matrix
   - Visualized misclassified images

4. ✅ **App Development**:
   - Built a simple Streamlit app (`app.py`)
   - Accepts image upload and displays prediction and confidence

---

## 🧠 Model Architecture

- **Base Model**: MobileNetV2 (frozen layers)
- **Custom Head**:
  - `GlobalAveragePooling2D`
  - `Dropout(0.2)`
  - `Dense(1, activation='sigmoid')`

---

## 💻 How to Run the App Locally

>  This app runs only on your local machine.

### ✅ Step 1: Install Requirements

'''bash
pip install streamlit tensorflow pillow numpy

### ✅ Step 2: Run the App

'''bash
streamlit run app.py


Then go to: http://localhost:8501 in your browser.

![Screenshot](https://github.com/gopikasabu25/Image_Classification/blob/main/sample.png)


![Screenshot](https://github.com/gopikasabu25/Image_Classification/blob/main/sample1.png)

----

## Project Structure:

cat_dog_classification/
├── app.py                # Streamlit app script (local only)
├── catdog_model.h5       # Trained model file
├── cat_dog_classifier.ipynb # Training + evaluation notebook
├── requirements.txt      # Python dependencies
└── README.md             # Project description

##📚 Credits

Dataset credit: Microsoft via [Kaggle](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)

Frameworks: TensorFlow, Keras, Streamlit
     
