import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec

RANDOM_STATE = 42
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSES = ["Joy", "Neutral", "Anger", "Surprise", "Sadness", "Fear"]
emotion_mapping = {
    0: "Joy", 1: "Joy", 2: "Anger", 3: "Anger", 4: "Joy", 5: "Joy",
    6: "Surprise", 7: "Surprise", 8: "Joy", 9: "Sadness", 10: "Anger",
    11: "Anger", 12: "Neutral", 13: "Joy", 14: "Fear", 15: "Joy",
    16: "Sadness", 17: "Joy", 18: "Joy", 19: "Fear", 20: "Joy",
    21: "Joy", 22: "Surprise", 23: "Joy", 24: "Sadness", 25: "Sadness",
    26: "Surprise", 27: "Neutral"
}
stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = re.findall(r"[a-z]+", text)
    tokens = [stemmer.stem(t) for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

def make_fallback_dataset(n_per_class=220):
    # This fallback keeps the app runnable without internet. Replace with GoEmotions download when online.
    templates = {
        "Joy": ["I feel happy grateful excited and proud today", "This is wonderful and makes me smile", "I love this good news so much"],
        "Neutral": ["I am reading the update and checking the information", "The post explains the situation in a simple way", "This is a normal comment about the topic"],
        "Anger": ["I am angry annoyed frustrated and upset about this", "This is unfair terrible and makes me mad", "I hate how rude and careless this feels"],
        "Surprise": ["I am shocked surprised and curious about what happened", "Wow I did not expect this sudden result", "This is unexpected confusing and surprising"],
        "Sadness": ["I feel sad lonely disappointed and tired", "This makes me cry and feel hopeless", "I miss the old days and feel down"],
        "Fear": ["I am scared worried nervous and afraid", "This situation feels dangerous and frightening", "I panic because I feel unsafe and anxious"],
    }
    rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    for label, phrases in templates.items():
        for i in range(n_per_class):
            base = phrases[i % len(phrases)]
            noise = rng.choice(["really", "very", "today", "now", "online", "people", "comment", "feel"], size=3, replace=True)
            rows.append({"text": f"{base} {' '.join(noise)}", "emotion": label})
    return pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

def load_goemotions_or_fallback():
    try:
        from datasets import load_dataset
        print("Loading GoEmotions simplified from HuggingFace...")
        raw = load_dataset("google-research-datasets/go_emotions", "simplified")
        frames = []
        for split in ["train", "validation", "test"]:
            df = pd.DataFrame(raw[split])
            df["emotion"] = df["labels"].apply(lambda labels: emotion_mapping.get(labels[0], "Neutral") if labels else "Neutral")
            frames.append(df[["text", "emotion"]])
        data = pd.concat(frames, ignore_index=True)
    except Exception as e:
        print("Could not download GoEmotions, using included fallback sample dataset.")
        print(e)
        data = make_fallback_dataset()
    data = data.dropna(subset=["text", "emotion"]).copy()
    data["clean_text"] = data["text"].apply(preprocess_text)
    data = data[data["clean_text"].str.len() > 0].reset_index(drop=True)
    return data

def average_word_vectors(token_lists, model, vector_size):
    vectors = []
    for tokens in token_lists:
        word_vecs = [model.wv[w] for w in tokens if w in model.wv]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(vector_size))
    return np.vstack(vectors)

def save_top_words(tfidf, X_train_tfidf, y_train):
    feature_names = np.array(tfidf.get_feature_names_out())
    rows = []
    for label in sorted(y_train.unique()):
        idx = np.where(y_train.values == label)[0]
        mean_scores = np.asarray(X_train_tfidf[idx].mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[-20:][::-1]
        for rank, feat_idx in enumerate(top_idx, 1):
            rows.append({"Emotion": label, "Rank": rank, "Word": feature_names[feat_idx], "TFIDF Score": float(mean_scores[feat_idx])})
    pd.DataFrame(rows).to_csv(os.path.join(MODEL_DIR, "top_words_by_class.csv"), index=False)

if __name__ == "__main__":
    data = load_goemotions_or_fallback()
    data.to_csv(os.path.join(DATA_DIR, "emotion_dataset.csv"), index=False)

    train_df, test_df = train_test_split(data, test_size=0.2, stratify=data["emotion"], random_state=RANDOM_STATE)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df["emotion"], random_state=RANDOM_STATE)

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(train_df["clean_text"])
    X_val_tfidf = tfidf.transform(val_df["clean_text"])
    X_test_tfidf = tfidf.transform(test_df["clean_text"])

    token_train = train_df["clean_text"].str.split().tolist()
    token_val = val_df["clean_text"].str.split().tolist()
    token_test = test_df["clean_text"].str.split().tolist()
    w2v = Word2Vec(sentences=token_train, vector_size=100, window=5, min_count=1, workers=2, sg=1, seed=RANDOM_STATE, epochs=10)
    X_train_w2v = average_word_vectors(token_train, w2v, 100)
    X_val_w2v = average_word_vectors(token_val, w2v, 100)
    X_test_w2v = average_word_vectors(token_test, w2v, 100)

    model_specs = {
        "Logistic Regression + TF-IDF": (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE), X_train_tfidf, X_val_tfidf, X_test_tfidf),
        "Linear SVM + TF-IDF": (LinearSVC(class_weight="balanced", random_state=RANDOM_STATE), X_train_tfidf, X_val_tfidf, X_test_tfidf),
        "Logistic Regression + Word2Vec": (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE), X_train_w2v, X_val_w2v, X_test_w2v),
        "Linear SVM + Word2Vec": (LinearSVC(class_weight="balanced", random_state=RANDOM_STATE), X_train_w2v, X_val_w2v, X_test_w2v),
    }

    results, predictions, model_files = [], {}, {}
    for name, (model, Xtr, Xv, Xte) in model_specs.items():
        print(f"Training {name}...")
        model.fit(Xtr, train_df["emotion"])
        val_pred = model.predict(Xv)
        test_pred = model.predict(Xte)
        precision, recall, f1, _ = precision_recall_fscore_support(test_df["emotion"], test_pred, average="macro", zero_division=0)
        file_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, file_name))
        model_files[name] = file_name
        predictions[name] = test_pred
        results.append({
            "Model": name,
            "Feature Method": "TF-IDF" if "TF-IDF" in name else "Word2Vec",
            "Val Accuracy": accuracy_score(val_df["emotion"], val_pred),
            "Test Accuracy": accuracy_score(test_df["emotion"], test_pred),
            "Precision": precision,
            "Recall": recall,
            "Macro F1": f1,
        })

    results_df = pd.DataFrame(results).sort_values("Macro F1", ascending=False)
    best_name = results_df.iloc[0]["Model"]
    best_pred = predictions[best_name]

    cm = confusion_matrix(test_df["emotion"], best_pred, labels=CLASSES)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(os.path.join(MODEL_DIR, "confusion_matrix.csv"))
    results_df.to_csv(os.path.join(MODEL_DIR, "model_results.csv"), index=False)
    data["emotion"].value_counts().rename_axis("Emotion").reset_index(name="Count").to_csv(os.path.join(MODEL_DIR, "class_distribution.csv"), index=False)
    test_df[["text", "clean_text", "emotion"]].to_csv(os.path.join(DATA_DIR, "test_data.csv"), index=False)
    save_top_words(tfidf, X_train_tfidf, train_df["emotion"])

    text_lengths = data.assign(TextLength=data["text"].str.split().str.len())[ ["emotion", "TextLength"] ]
    text_lengths.to_csv(os.path.join(MODEL_DIR, "text_length_distribution.csv"), index=False)

    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    w2v.save(os.path.join(MODEL_DIR, "word2vec.model"))
    metadata = {
        "project_title": "Social Media Emotion Analyzer",
        "dataset": "GoEmotions simplified, fallback sample included for offline runs",
        "classes": CLASSES,
        "best_model_name": best_name,
        "best_model_file": model_files[best_name],
        "model_files": model_files,
        "preprocessing": ["lowercase", "remove URLs", "remove special characters/numbers", "tokenization", "stopword removal", "stemming"],
        "feature_methods": ["TF-IDF", "Word2Vec average embeddings"]
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, "metadata.joblib"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
        f.write(classification_report(test_df["emotion"], best_pred, zero_division=0))
    print("Training complete. Best model:", best_name)
