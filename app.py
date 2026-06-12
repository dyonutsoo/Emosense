import os
import re
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec

st.set_page_config(page_title="Social Media Emotion Analyzer", page_icon="🐦", layout="wide")
DATA_DIR = "data"
MODEL_DIR = "models"
stemmer = PorterStemmer()

@st.cache_data
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = re.findall(r"[a-z]+", text)
    tokens = [stemmer.stem(t) for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

@st.cache_resource
def load_artifacts():
    metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    best_model = joblib.load(os.path.join(MODEL_DIR, metadata["best_model_file"]))
    results = pd.read_csv(os.path.join(MODEL_DIR, "model_results.csv"))
    cm = pd.read_csv(os.path.join(MODEL_DIR, "confusion_matrix.csv"), index_col=0)
    class_dist = pd.read_csv(os.path.join(MODEL_DIR, "class_distribution.csv"))
    top_words = pd.read_csv(os.path.join(MODEL_DIR, "top_words_by_class.csv"))
    lengths = pd.read_csv(os.path.join(MODEL_DIR, "text_length_distribution.csv"))
    data = pd.read_csv(os.path.join(DATA_DIR, "emotion_dataset.csv"))
    return metadata, tfidf, best_model, results, cm, class_dist, top_words, lengths, data

try:
    metadata, tfidf, model, results_df, cm_df, class_dist_df, top_words_df, lengths_df, data_df = load_artifacts()
except Exception as e:
    st.error("Model files are missing. Run `python train_models.py` first, then run `streamlit run app.py`.")
    st.exception(e)
    st.stop()

def prediction_scores(model, X):
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[0]
    elif hasattr(model, "decision_function"):
        raw = np.ravel(model.decision_function(X))
        exp = np.exp(raw - np.max(raw))
        scores = exp / exp.sum()
    else:
        pred = model.predict(X)[0]
        scores = np.array([1.0 if c == pred else 0.0 for c in classes])
    return pd.DataFrame({"Emotion": classes, "Confidence": scores}).sort_values("Confidence", ascending=False)

def input_influential_words(clean_text):
    if not clean_text:
        return pd.DataFrame(columns=["Word", "TF-IDF Weight"])
    X = tfidf.transform([clean_text])
    arr = X.toarray().ravel()
    names = np.array(tfidf.get_feature_names_out())
    idx = arr.argsort()[-10:][::-1]
    rows = [{"Word": names[i], "TF-IDF Weight": arr[i]} for i in idx if arr[i] > 0]
    return pd.DataFrame(rows)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose page", ["Home/About", "Text Analyzer", "Data Explorer", "Visualizations", "Model Info"])
st.sidebar.info(f"Best model: {metadata['best_model_name']}")

if page == "Home/About":
    st.title("🐦 Social Media Emotion Analyzer")
    st.write("This NLP application classifies social media text into emotion categories and shows model and data insights.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset samples", f"{len(data_df):,}")
    c2.metric("Emotion classes", len(metadata["classes"]))
    c3.metric("Models compared", len(results_df))
    st.subheader("Problem solved")
    st.write("Social media posts can contain emotions such as joy, sadness, anger, fear, surprise, or neutral tone. This app helps users enter a text and instantly predict the emotion.")
    st.subheader("How to use")
    st.write("Open the Text Analyzer page, paste a sentence or social media comment, then click Analyze Text. The app will show the predicted emotion, confidence score, and important words.")
    st.subheader("Team members")
    st.write("Laila, Wajee, Shree, Qistina")

elif page == "Text Analyzer":
    st.title("🔍 Text Analyzer")
    text = st.text_area("Enter social media text", "I am so happy and grateful for this amazing day!", height=150)
    if st.button("Analyze Text"):
        clean = preprocess_text(text)
        X = tfidf.transform([clean])
        pred = model.predict(X)[0]
        scores_df = prediction_scores(model, X)
        st.success(f"Predicted emotion: **{pred}**")
        st.metric("Confidence", f"{scores_df.iloc[0]['Confidence']*100:.2f}%")
        st.caption(f"Preprocessed text: `{clean}`")
        fig = px.bar(scores_df, x="Emotion", y="Confidence", title="Confidence Score by Emotion", text_auto=".2%")
        st.plotly_chart(fig, use_container_width=True)
        words_df = input_influential_words(clean)
        st.subheader("Words that influenced the prediction")
        if len(words_df):
            st.dataframe(words_df, use_container_width=True)
        else:
            st.warning("No known important words found. Try entering a longer sentence.")

elif page == "Data Explorer":
    st.title("📊 Data Explorer")
    st.write("Sample dataset and basic statistics.")
    st.dataframe(data_df[["text", "emotion", "clean_text"]].head(100), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", f"{len(data_df):,}")
    c2.metric("Missing text", int(data_df["text"].isna().sum()))
    c3.metric("Average text length", f"{data_df['text'].str.split().str.len().mean():.1f} words")
    fig_pie = px.pie(class_dist_df, names="Emotion", values="Count", title="Class Distribution Pie Chart")
    st.plotly_chart(fig_pie, use_container_width=True)

elif page == "Visualizations":
    st.title("📈 Visualizations")
    st.write("Minimum 5 required charts are included: word cloud, class distribution, confusion matrix, model comparison, and top words. Text length distribution is an extra chart.")

    st.subheader("1. Word Cloud")
    selected_class = st.selectbox("Choose emotion for word cloud", ["All"] + metadata["classes"])
    if selected_class == "All":
        text_blob = " ".join(data_df["clean_text"].dropna().astype(str))
    else:
        text_blob = " ".join(data_df.loc[data_df["emotion"] == selected_class, "clean_text"].dropna().astype(str))
    wc = WordCloud(width=1000, height=420, background_color="white", max_words=120).generate(text_blob)
    fig_wc, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)

    st.subheader("2. Class Distribution")
    fig_dist = px.bar(class_dist_df, x="Emotion", y="Count", title="Number of Samples by Emotion", text_auto=True)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("3. Confusion Matrix")
    fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", title=f"Confusion Matrix for {metadata['best_model_name']}")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("4. Model Comparison")
    metrics = ["Test Accuracy", "Precision", "Recall", "Macro F1"]
    melted = results_df.melt(id_vars=["Model", "Feature Method"], value_vars=metrics, var_name="Metric", value_name="Score")
    fig_model = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group", title="Model Performance Comparison")
    st.plotly_chart(fig_model, use_container_width=True)

    st.subheader("5. Top 20 Words by Emotion")
    emotion = st.selectbox("Choose emotion for top words", metadata["classes"])
    top20 = top_words_df[top_words_df["Emotion"] == emotion].head(20)
    fig_top = px.bar(top20, x="TFIDF Score", y="Word", orientation="h", title=f"Top 20 Important Words: {emotion}")
    fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("Extra: Text Length Distribution")
    fig_len = px.histogram(lengths_df, x="TextLength", color="emotion", nbins=30, title="Text Length Distribution by Emotion")
    st.plotly_chart(fig_len, use_container_width=True)

elif page == "Model Info":
    st.title("🤖 Model Info")
    st.subheader("NLP Pipeline")
    st.write("Preprocessing steps: lowercase, remove URLs, remove special characters/numbers, tokenization, stopword removal, and stemming.")
    st.write("Feature extraction methods: TF-IDF and Word2Vec average embeddings.")
    st.subheader("Models trained and compared")
    st.dataframe(results_df, use_container_width=True)
    st.subheader("Best model")
    st.success(metadata["best_model_name"])
    st.subheader("Training details")
    st.json(metadata)
    report_path = os.path.join(MODEL_DIR, "classification_report.txt")
    if os.path.exists(report_path):
        st.text(open(report_path, encoding="utf-8").read())
