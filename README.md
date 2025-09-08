# 🏰 Signs of the Palace

**Signs of the Palace** is a deep learning-based **sign language recognition system** that classifies palace-related objects and gestures from short video clips.  
It uses a **CNN + LSTM model** to learn spatio-temporal features and predict the correct class among 29 custom palace-related categories.  

---

## ✨ Features
- 🎥 Recognizes palace-themed sign language gestures from video clips.
- 🧠 Uses **CNN + LSTM** for spatio-temporal sequence learning.
- 🏛️ Trained on a **custom dataset** with 29 palace-related classes.
- 📊 Supports classification from **16-frame video sequences**.
- 🌍 Aims to preserve cultural heritage with modern AI.

---

## 📂 Dataset
The dataset contains **29 palace-related classes**, each with **75 videos**.  
Examples of classes:
- Mysore Palace  
- Ambari  
- King’s Crown  
- Sword  
- Elephant  
- Horse  
- Soldiers  
- Queen  
- Minister (Mantri)  
- Chamundeshwari  
- Main Door, North Gate, South Gate  
- Paintings, Temples, Canon, Shield, Flag, and more  

Each video is processed into **8-frame clips** for training and prediction.

---

## 🏗️ Model
- **Architecture**: CNN (spatial feature extraction) + LSTM (temporal sequence learning)  
- **Input**: 16 frames per video clip  
- **Output**: One of 29 palace-related classes  
- **Frameworks Used**: TensorFlow / Keras  

---

## 🚀 Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/signs-of-the-palace.git
   cd signs-of-the-palace
