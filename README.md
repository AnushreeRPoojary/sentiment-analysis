# 🐦 Twitter Sentiment Analysis using BERT

Fine-tuned BERT model for classifying tweets into 4 sentiment categories — Positive, Negative, Neutral, and Irrelevant — achieving **91% accuracy** on 73K+ tweets.

---

## 📌 Project Overview

This project fine-tunes `bert-base-uncased` on the Twitter Entity Sentiment Analysis dataset. It covers the full ML pipeline: data preprocessing, EDA, model training, evaluation, and visualization.

---

---

## 👩‍💻 Author

| | |
|---|---|
| **Name** | Anushree R |
| **Email** | anushreerpoojary2611@gmail.com |
| **GitHub** | [github.com/AnushreeRPoojary](https://github.com/AnushreeRPoojary) |

---

## 🗂️ Dataset

- **Source:** Twitter Entity Sentiment Analysis (Kaggle)
- **File:** `twitter_training.csv`
- **Size:** 73,140 tweets
- **Classes:** Positive, Negative, Neutral, Irrelevant

---

## 🧠 Model

- **Base Model:** `bert-base-uncased` (HuggingFace Transformers)
- **Task:** Multi-class Sequence Classification (4 classes)
- **Optimizer:** AdamW with weight decay
- **Scheduler:** Linear warmup + decay
- **Max Token Length:** 128
- **Batch Size:** 32
- **Epochs:** 3
- **Learning Rate:** 2e-5

---

## 📊 Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1     | 0.8990    | 63.33%    | 0.5890   | 77.94%  |
| 2     | 0.3961    | 86.01%    | 0.3250   | 89.07%  |
| 3     | 0.1827    | 93.72%    | 0.2843   | **91.30%** |

### Per-Class Metrics (Best Model)

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Irrelevant | 0.90      | 0.91   | 0.90     |
| Negative   | 0.93      | 0.93   | 0.93     |
| Neutral    | 0.91      | 0.90   | 0.90     |
| Positive   | 0.90      | 0.92   | 0.91     |
| **Overall**| **0.91**  | **0.91**| **0.91** |

---

## 📁 Project Structure

```
├── twitter_sentiment_bert.ipynb   # Main Colab notebook         # ipynb file downloaded without outputs
├── twitter_training.csv           # Dataset (upload to Colab)      
├── best_bert_model.pt             # Saved best model weights    #since its downloaded from colab bert_model not in my  project structure but can be added here
├── eda_analysis.png               # EDA visualizations
├── training_curves.png            # Loss & accuracy curves
├── confusion_matrix.png           # Confusion matrix heatmap    #all outputs  png files are in pdf of my project structure  
└── README.md
```

---

## ⚙️ Setup & Usage

### Run on Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Run the install cell:
```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn pandas numpy tqdm
```
3. Upload `twitter_training.csv` when prompted
4. Run all cells sequentially

> **Note:** GPU (CUDA) recommended — training takes ~20 min per epoch on T4.

---

## 📈 Visualizations

- **Class Distribution** — bar chart of sentiment counts
- **Tweet Length Distribution** — word count histogram per class
- **Sentiment by Topic** — top 8 topics breakdown
- **Training Curves** — loss & accuracy per epoch
- **Confusion Matrix** — heatmap of predictions vs actual
- **Per-Class Metrics** — precision, recall, F1 bar chart

---

## 🛠️ Tech Stack

- Python 3
- PyTorch
- HuggingFace Transformers
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- Google Colab (GPU)

---

## 👩‍💻 Author

**Anushree R Poojary**  
[GitHub](https://github.com/AnushreeRPoojary)

*Built by Anushree R — AI & ML Engineer*  
*B.E. Artificial Intelligence & Machine Learning, SMVITM Udupi*
