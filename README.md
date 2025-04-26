# 📰 Fake News Detection Web App

A Machine Learning-powered web application built using **Streamlit** that identifies whether a news article is **real** or **fake**. This project was developed by **Riya Maurya** as part of a hands-on learning initiative in Machine Learning and Natural Language Processing.

## 🔍 Features

- 🧠 Predicts real vs fake news using a trained Random Forest model.
- 📊 Confidence score visualization using a pie chart.
- ☁️ Generates word clouds for real and fake news.
- 📝 Try with custom input or predefined real/fake examples.

## 🛠️ Tech Stack

- **Python**
- **Scikit-learn**
- **Streamlit**
- **TF-IDF Vectorizer**
- **Pickle** for model storage
- **Matplotlib** and **WordCloud** for visualization

## 🚀 Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector
   
2. Install dependencies:

pip install -r requirements.txt

4. Run the app:
   
streamlit run app.py

## 🧠 ML Model
Model: Random Forest Classifier

Vectorization: TF-IDF

The model and vectorizer were pre-trained and saved as model.pkl and vectorizer.pkl using pickle.

## 📁 Project Structure

├── app.py               # Streamlit app
├── model.pkl            # Trained ML model
├── vectorizer.pkl       # TF-IDF vectorizer
├── requirements.txt     # Python dependencies
├── README.md            # This file!

## ✨ Demo
Deploy on Streamlit Cloud for free!
LINK: https://fake-news-detector-using-ml-qbv32pbpnmgb9ty5upv6gj.streamlit.app/

## 📬 Contact
Created by Riya Maurya – feel free to reach out for feedback or collaboration!

