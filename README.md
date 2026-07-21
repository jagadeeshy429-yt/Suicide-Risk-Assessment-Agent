# 🧠 Suicide Risk Assessment Agent

An AI-powered Machine Learning application that analyzes text and predicts the risk of suicide based on Natural Language Processing (NLP) techniques. The project is designed for educational and research purposes to demonstrate how machine learning can assist in identifying potentially high-risk text. It is **not** a substitute for professional mental health assessment or emergency services. :contentReference[oaicite:0]{index=0}

---

## 📌 Features

- Predicts suicide risk from user-entered text.
- NLP-based text preprocessing.
- TF-IDF feature extraction.
- Machine Learning classification model.
- Simple and user-friendly interface.
- Fast predictions.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Joblib
- Streamlit (for web application)

---

## 📂 Project Structure

```
Suicide-Risk-Assessment-Agent/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── Suicide_Detection.csv
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/jagadeeshy429-yt/Suicide-Risk-Assessment-Agent.git
```

Move into the project directory:

```bash
cd Suicide-Risk-Assessment-Agent
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

If using Streamlit:

```bash
streamlit run app.py
```

---

## 📊 Dataset

The model is trained on a suicide detection text dataset containing social media posts labeled as:

- Suicide
- Non-Suicide

---

## 🤖 Machine Learning Pipeline

1. Load Dataset
2. Data Cleaning
3. Text Preprocessing
4. TF-IDF Vectorization
5. Model Training
6. Model Evaluation
7. Save Model
8. Predict User Input

---

## 📈 Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## ⚠️ Disclaimer

This project is intended **only for educational and research purposes**. It must **not** be used as a medical or clinical diagnostic tool. Predictions produced by the model should never replace assessment or intervention by qualified mental health professionals. :contentReference[oaicite:1]{index=1}

---

## 👨‍💻 Author

**Jagadeesh Chinni**

GitHub: https://github.com/jagadeeshy429-yt

---

## 📜 License

This project is released under the MIT License.
