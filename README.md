# 📩 SMS Spam Classification  
**Applied Machine Learning Coursework**

---

## 🔍 Project Overview
This project focuses on **SMS text classification** as part of the **Applied Machine Learning** module. The objective is to automatically classify short text messages as either **spam** or **ham (legitimate)** using supervised machine learning techniques. SMS spam detection is a practical real-world problem that demonstrates how machine learning can be applied to natural language processing (NLP) tasks.

The project implements and compares **two different modeling approaches** using a consistent preprocessing and evaluation pipeline:
- **Multinomial Naive Bayes (from scratch)** combined with **TF-IDF feature extraction**, providing a fast, interpretable, and effective baseline for text classification.
- **DistilBERT (fine-tuned)**, a transformer-based language model that captures contextual and semantic information within messages, enabling improved performance on more ambiguous or complex text.

To ensure a fair comparison, both models use the **same stratified 80/10/10 train–validation–test split** and are evaluated using metrics suitable for imbalanced datasets, including **precision, recall, F1-score, confusion matrices, ROC curves, and Precision–Recall curves**. Sample predictions are also generated to demonstrate real-world inference and support error and ethical analysis.

---

## 🛠️ Technologies Used
- **Python**
- **Pandas, NumPy** – data handling and preprocessing  
- **scikit-learn** – TF-IDF feature extraction and evaluation metrics  
- **Matplotlib** – data visualisation  
- **PyTorch** – deep learning framework  
- **Hugging Face Transformers** – DistilBERT fine-tuning  
- **Jupyter Notebook** – experimentation and analysis  

---

## 📁 Project Structure
.
├── README.md
├── data
│   ├── cleaned
│   ├── raw
│   │   └── spam-sms.csv
│   └── splits
├── models
│   ├── distilbert
│   └── naive_bayes
├── notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_naive_bayes.ipynb
│   ├── 03_distilbert.ipynb
│   └── 04_model_comparison.ipynb
├── outputs
│   ├── plots
│   └── predictions
└── src
├── evaluation.py
├── preprocessing.py
├── train_bert.py
└── train_nb.py

---

## ✅ Key Learning Outcomes
- Applied supervised machine learning to a real-world text classification problem  
- Implemented a probabilistic classifier (**Naive Bayes**) from scratch  
- Fine-tuned a pretrained transformer model (**DistilBERT**)  
- Evaluated models using appropriate metrics for imbalanced datasets  
- Analysed model behaviour, limitations, and ethical considerations  

---

## 📌 Notes
This project was developed for academic purposes as part of the **Applied Machine Learning** module and emphasizes reproducibility, fair model comparison, and practical evaluation of NLP models.

---