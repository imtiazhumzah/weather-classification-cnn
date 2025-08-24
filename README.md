
# Weather Classification using Deep Learning

## Project Overview
This project focuses on **classifying weather conditions (e.g., cloudy, sunny, rainy, snowy, etc.)** from image datasets using **Convolutional Neural Networks (CNNs)** and **Transfer Learning (MobileNetV2)**.  
The goal is to develop a model that can recognize weather patterns from raw images, enabling applications in **smart cities, autonomous vehicles, and environmental monitoring**.

---

## Key Features
- **Dataset Preprocessing**: 
  - Splits dataset into **train (70%)**, **validation (20%)**, and **test (10%)** sets.
  - Performs **image augmentation** for robust training.
- **Visualization**: Displays sample images per class & dataset distribution.
- **Models Used**:
- **Custom CNN** (built from scratch with Conv2D + MaxPooling).
- **Transfer Learning with MobileNetV2** for improved performance.
- **Evaluation**:
  - Accuracy & loss curves
  - Classification report & confusion matrix
  - Misclassified image visualization
- **Model Improvement**:
  - Fine-tuned **MobileNetV2** with dropout regularization.
  - Achieved higher validation accuracy compared to baseline CNN.

---

## Tech Stack
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow / Keras
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, PIL
- **Utilities**: tqdm, shutil, os

---

## Project Workflow
1. **Dataset Loading & Visualization**
   - Mounted dataset from Google Drive
   - Visualized random samples from each class

2. **Data Preprocessing**
   - Split into train/val/test
   - Applied augmentation: rotation, zoom, flips, brightness shifts

3. **Model Building**
   - Baseline CNN with Conv2D + Dense layers
   - MobileNetV2 transfer learning pipeline

4. **Training & Evaluation**
   - Trained models for 5–10 epochs
   - Compared validation accuracy & loss
   - Generated confusion matrix & misclassified image plots

5. **Model Improvement**
   - Fine-tuned MobileNetV2
   - Applied dropout layers to reduce overfitting

---

## 📊 Results
- **Baseline CNN**: Moderate accuracy (~70–75%) with some overfitting.
- **MobileNetV2 Transfer Learning**: Improved accuracy (~85–90%) and better generalization.
- Confusion matrix shows strong performance across most classes but minor misclassifications in visually similar weather conditions.

---

## Sample Visualizations
- Training vs Validation Accuracy & Loss  
- Confusion Matrix  
- Misclassified Images with Predicted vs Actual Labels  

*(<img width="647" height="237" alt="Acc before improv" src="https://github.com/user-attachments/assets/4ca1f5cf-5a39-4b35-b73a-88c6dec003cb" />
)*
*(<img width="754" height="351" alt="Misclassified imgs" src="https://github.com/user-attachments/assets/adee1495-f166-4aa9-badd-767d41b9d05f" />
)*

---

## Future Work
- Experiment with **other pre-trained models** (EfficientNet, ResNet50).
- Deploy as a **Streamlit/Flask Web App** for real-time predictions.
- Extend dataset with more diverse weather conditions.
- Optimize training with **hyperparameter tuning**.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/imtiazhumzah/weather-classification-cnn.git
   cd weather-classification-cnn

