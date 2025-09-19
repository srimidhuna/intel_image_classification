🖼️ Intel Image Classification with Deep Learning

📌 Project Overview

This project focuses on classifying natural scenes into six categories from the Intel Image dataset.
The trained model (my_model.h5) can recognize:
🏠 Buildings | 🌲 Forest | ❄️ Glacier | ⛰️ Mountain | 🌊 Sea | 🛣️ Street

🎯 Objectives

Preprocess and augment the dataset.

Train a deep learning model (CNN) for scene recognition.

Compare performance using multiple evaluation metrics.

Visualize the training process and classification results.

🛠️ Tech Stack

Backend: TensorFlow / Keras

Visualization: Matplotlib, Seaborn

Notebook: Jupyter (Untitled6.ipynb)

Database (for logging): SQLite (lightweight and easy)

📂 Dataset

Source: Intel Image Classification Dataset

Classes:

Buildings

Forest

Glacier

Mountain

Sea

Street

🔄 Workflow

Data Preprocessing

Resize images

Normalize pixel values

Augment dataset with flips, rotations, and zoom

Model Training

CNN model with Conv2D, MaxPooling, Dropout

Optimizer: Adam

Loss: Categorical Crossentropy

Evaluation Metrics

Accuracy

Precision, Recall, F1-score


📊 Visual Results
Training Performance

Loss Curve

Sample Predictions

🚀 How to Run
1️⃣ Clone the repo
git clone https://github.com/your-username/intel-image-classification.git
cd intel-image-classification

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the notebook
jupyter notebook Untitled6.ipynb

4️⃣ Use the trained model
from tensorflow.keras.models import load_model
model = load_model("my_model.h5")

📌 Future Improvements

Add Transfer Learning (ResNet, VGG16, EfficientNet).

Deploy the model with Streamlit or Flask.

Improve accuracy with hyperparameter tuning.

✨ Example Predictions

🌟 This project demonstrates how deep learning can classify natural scenes into multiple categories, making it useful for geospatial analytics, tourism apps, and environmental monitoring.