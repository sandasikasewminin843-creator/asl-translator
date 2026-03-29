# ASL Translator 🤟

A real-time American Sign Language (ASL) translator that detects hand signs using a webcam or phone camera and converts them into text. Built as a final project for my AI/ML course.

---

## 📌 What It Does

- Detects ASL hand signs (A–Z + SPACE) in real time using a webcam or phone camera
- Converts detected signs into text letter by letter
- Supports image upload for single-image sign prediction
- Web-based interface accessible from any browser

---

## ✨ Features

- 🎥 Live camera feed with real-time sign detection
- ⏱ Auto-add letters by holding a sign steady for 2.5 seconds
- 📷 Camera on/off toggle and camera selection (built-in or phone camera)
- 🔄 Camera flip/mirror option
- 🖼 Image upload to predict a sign from a photo
- 🌙 Dark mode / ☀️ Light mode toggle
- ✍️ Text output area that works like a keyboard

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Hand Detection | MediaPipe Hand Landmarker |
| Classification | Random Forest (scikit-learn) |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| Camera | OpenCV + Iriun Webcam |

---

## 🚀 How to Run Locally

**1. Clone the repository**
```
git clone https://github.com/sandasikasewminin843-creator/asl-translator.git
cd asl-translator
```

**2. Create a virtual environment and install dependencies**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run the app**
```
python app.py
```

**4. Open your browser at**
```
http://localhost:5000
```

---

## 📁 Project Structure
```
asl-translator/
├── app.py                  # Flask backend
├── index.html              # Frontend web interface
├── inference_classifier.py # Standalone OpenCV detector
├── train_classifier.py     # Model training script
├── create_dataset.py       # Landmark extraction from images
├── collect_imgs.py         # Image collection script
├── model.p                 # Trained Random Forest model
├── data.pickle             # Extracted hand landmark dataset
└── requirements.txt        # Python dependencies
```

---

## 🧠 How It Works

1. **Data Collection** — Hand sign images collected using `collect_imgs.py` via webcam
2. **Landmark Extraction** — MediaPipe detects 21 hand landmarks per image, saved to `data.pickle`
3. **Model Training** — A Random Forest classifier trained on the landmark coordinates
4. **Inference** — Live camera frames are processed by MediaPipe, landmarks fed to the model, predicted letter displayed on screen

---

## 🔮 Future Improvements

- Add number signs (0–9)
- Improve accuracy for similar signs (E/O, M/N)
- Add NLP to suggest complete words from partial input
- Mobile app version
- Support for two-handed signs

---

## 👤 Author

**Sandasika Sewminin**  
```

---

To add it to GitHub, just create the file locally and push:
```
git add README.md
git commit -m "Add README"
git push origin main

