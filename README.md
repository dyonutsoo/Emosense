# Social Media Emotion Analyzer

This is a full SAIA 2163 Final Project Streamlit app for Theme 5: Social Media Emotion Analyzer.

## Features
- Text input and instant emotion prediction
- Confidence score for each emotion
- Influential words from TF-IDF
- Data explorer with dataset statistics
- 5 required visualizations:
  1. Word cloud
  2. Class distribution
  3. Confusion matrix
  4. Model comparison
  5. Top 20 words / feature importance
- Extra visualization: text length distribution
- Four models trained and compared:
  - Logistic Regression + TF-IDF
  - Linear SVM + TF-IDF
  - Logistic Regression + Word2Vec
  - Linear SVM + Word2Vec

## Setup
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## GitHub structure
```text
app.py
train_models.py
requirements.txt
README.md
data/
models/
notebooks/
docs/
.gitignore
```

## Dataset
The training script tries to download GoEmotions simplified from HuggingFace. If internet is unavailable, it uses the included fallback sample generator so the app can still run.

## Team members
Laila, Wajee, Shree, Qistina
