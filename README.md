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
Accuracy 


<img width="567" height="432" alt="image" src="https://github.com/user-attachments/assets/48578838-6e1d-4560-8a38-d7c64f4e2bce" />


Loss Curve

<img width="846" height="547" alt="image" src="https://github.com/user-attachments/assets/3045d28c-7b14-48b4-8c93-df5f64968796" />


Sample Predictions

<img width="1900" height="825" alt="image" src="https://github.com/user-attachments/assets/c7211a67-2388-464f-9aa8-3f15b95cbad8" />

🚀 How to Run
1️⃣ Clone the repo
git clone https://github.com/srimidhuna/intel_image_classification
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

Deploy the model with Gradio.

Improve accuracy with hyperparameter tuning.

✨ Example Predictions

🌟 This project demonstrates how deep learning can classify natural scenes into multiple categories, making it useful for geospatial analytics, tourism apps, and environmental monitoring.

APP LINK: https://huggingface.co/spaces/srimidhuna/intel_image_classification
