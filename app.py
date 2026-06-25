import os
import re
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

st.set_page_config(
    page_title="Social Media Emotion Analyzer",
    page_icon="\U0001F3AD",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data"
MODEL_DIR = "models"
stemmer = PorterStemmer()

NEGATION_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "neither", "cannot", "cant",
    "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "werent", "wont",
    "wouldnt", "shouldnt", "couldnt", "without", "hardly", "aint",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS

EMOTION_STYLE = {
    "Joy": ("\U0001F60A", "#FFC857"),
    "Neutral": ("\U0001F610", "#9AA5B1"),
    "Anger": ("\U0001F620", "#FF5C5C"),
    "Surprise": ("\U0001F632", "#4ECDC4"),
    "Sadness": ("\U0001F622", "#5B8DEF"),
    "Fear": ("\U0001F628", "#B57BFF"),
}
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#7C5CFF"
CONF_THRESHOLD = 0.50

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(1200px 600px at 10% -10%, #1b2030 0%, #0E1117 55%); }
    .block-container { padding-top: 5rem; padding-bottom: 3rem; max-width: 1200px; }
    .hero {
        background: #5B57D1;
        border-radius: 20px; padding: 34px 38px; margin-bottom: 26px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.30);
    }
    .hero h1 { color: #fff; font-size: 2.2rem; margin: 0 0 6px 0; font-weight: 800; }
    .hero p  { color: rgba(255,255,255,0.92); font-size: 1.02rem; margin: 0; }
    .card {
        background: #161B26; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 22px 24px; height: 100%;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .metric-card {
        background: linear-gradient(160deg, #1a2030 0%, #141925 100%);
        border: 1px solid rgba(124,92,255,0.18); border-radius: 16px;
        padding: 18px 20px; text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 800; color: #fff; }
    .metric-card .label { font-size: 0.85rem; color: #9AA5B1; letter-spacing: .04em; text-transform: uppercase; }
    .result-card {
        background: #161B26; border-radius: 18px; padding: 26px 28px;
        border: 1px solid rgba(255,255,255,0.07); text-align: center;
    }
    .result-emoji { font-size: 3.4rem; line-height: 1; }
    .result-label { font-size: 1.7rem; font-weight: 800; margin-top: 6px; }
    .result-conf  { color: #9AA5B1; margin-top: 2px; }
    .pill {
        display:inline-block; padding: 4px 12px; border-radius: 999px;
        background: rgba(124,92,255,0.15); color: #c9bcff; font-size: .8rem;
        border: 1px solid rgba(124,92,255,0.3); margin: 2px;
    }
    .stButton > button {
        background: #5B57D1; color: #fff;
        border: none; border-radius: 12px; padding: 0.6rem 1.4rem;
        font-weight: 700; transition: transform .08s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); filter: brightness(1.07); }
    section[data-testid="stSidebar"] { background: #0b0e15; border-right: 1px solid rgba(255,255,255,0.05); }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = re.findall(r"[a-z]+", text)
    tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


@st.cache_resource
def load_artifacts():
    metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    tfidf_models = {}
    for name, fname in metadata["model_files"].items():
        if "tf_idf" in fname or "tfidf" in fname:
            try:
                tfidf_models[name] = joblib.load(os.path.join(MODEL_DIR, fname))
            except Exception:
                pass
    results = pd.read_csv(os.path.join(MODEL_DIR, "model_results.csv"))
    cm = pd.read_csv(os.path.join(MODEL_DIR, "confusion_matrix.csv"), index_col=0)
    class_dist = pd.read_csv(os.path.join(MODEL_DIR, "class_distribution.csv"))
    top_words = pd.read_csv(os.path.join(MODEL_DIR, "top_words_by_class.csv"))
    lengths = pd.read_csv(os.path.join(MODEL_DIR, "text_length_distribution.csv"))
    data = pd.read_csv(os.path.join(DATA_DIR, "emotion_dataset.csv"))
    return metadata, tfidf, tfidf_models, results, cm, class_dist, top_words, lengths, data


def detect_and_translate(text: str):
    try:
        from langdetect import detect
        lang = detect(text)
    except Exception:
        return text, "en"
    if lang == "en":
        return text, "en"
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated or text, lang
    except Exception:
        return text, lang


TRANSFORMER_DISPLAY_NAME = "⚡ DistilRoBERTa (Transformer)"
_HF_MODEL = "j-hartmann/emotion-english-distilroberta-base"
_TRANSFORMER_LABEL_MAP = {
    "anger": "Anger", "disgust": "Anger", "fear": "Fear",
    "joy": "Joy", "neutral": "Neutral", "sadness": "Sadness", "surprise": "Surprise",
}


@st.cache_resource
def load_transformer():
    try:
        from transformers import pipeline as hf_pipeline
        return hf_pipeline(
            "text-classification", model=_HF_MODEL,
            top_k=None, device=-1,
        )
    except Exception:
        return None


def transformer_scores(pipe, text: str) -> pd.DataFrame:
    raw = pipe(text, truncation=True, max_length=512)[0]
    if isinstance(raw, dict):
        raw = [raw]
    agg: dict = {}
    for r in raw:
        label = _TRANSFORMER_LABEL_MAP.get(r["label"], "Neutral")
        agg[label] = agg.get(label, 0.0) + r["score"]
    total = sum(agg.values()) or 1.0
    df = pd.DataFrame(
        [{"Emotion": k, "Confidence": v / total} for k, v in agg.items()]
    ).sort_values("Confidence", ascending=False)
    return df.reset_index(drop=True)


try:
    (metadata, tfidf, tfidf_models, results_df, cm_df, class_dist_df,
     top_words_df, lengths_df, data_df) = load_artifacts()
except Exception as e:
    st.error("Model files are missing. Run `python train_models.py` first, then `streamlit run app.py`.")
    st.exception(e)
    st.stop()


def style_fig(fig, height=380):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
        font=dict(color="#E6E9EF"),
        title_font=dict(size=16),
    )
    return fig


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
    rows = [{"Word": names[i], "TF-IDF Weight": round(float(arr[i]), 4)} for i in idx if arr[i] > 0]
    return pd.DataFrame(rows)


st.sidebar.markdown("## \U0001F3AD Emotion Analyzer")
page = st.sidebar.radio(
    "Navigate",
    ["Text Analyzer", "Overview", "Data Explorer", "Visualizations", "Model Info"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Best model**")
st.sidebar.markdown(f"<span class='pill'>{metadata['best_model_name']}</span>", unsafe_allow_html=True)
st.sidebar.caption(f"{len(data_df):,} samples - {len(metadata['classes'])} emotions")


def metric_row(items):
    cols = st.columns(len(items))
    for col, (val, label) in zip(cols, items):
        col.markdown(
            f"<div class='metric-card'><div class='value'>{val}</div>"
            f"<div class='label'>{label}</div></div>",
            unsafe_allow_html=True,
        )


if page == "Text Analyzer":
    st.markdown(
        "<div class='hero'><h1>\U0001F3AD Social Media Emotion Analyzer</h1>"
        "<p>Paste any social media post or comment — the model predicts the emotion behind it with confidence scores and the key words that drove the decision.</p></div>",
        unsafe_allow_html=True,
    )

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    if "auto_analyze" not in st.session_state:
        st.session_state["auto_analyze"] = False

    left, right = st.columns([1.15, 1])
    with left:
        model_names = list(tfidf_models.keys()) + [TRANSFORMER_DISPLAY_NAME]
        default_idx = (model_names.index(metadata["best_model_name"])
                       if metadata["best_model_name"] in model_names else 0)
        model_choice = st.selectbox("Model", model_names, index=default_idx)

        st.markdown(
            "<p style='margin:10px 0 4px;font-size:0.78rem;color:#9AA5B1;"
            "font-weight:600;letter-spacing:.06em;text-transform:uppercase;'>"
            "💡 Try an example</p>",
            unsafe_allow_html=True,
        )
        EXAMPLES = [
            ("\U0001F60A Joy",      "I am so happy and grateful for this amazing day!"),
            ("\U0001F620 Anger",    "This is absolutely infuriating, I am so done with this."),
            ("\U0001F622 Sadness",  "I miss them so much, everything feels empty now."),
            ("\U0001F632 Surprise", "Wait, I genuinely did not see that coming at all!"),
            ("\U0001F628 Fear",     "I am terrified of what might happen, I cannot stop worrying."),
            ("\U0001F610 Neutral",  "The team submitted the weekly report for review today."),
        ]
        for row_start in range(0, len(EXAMPLES), 3):
            cols = st.columns(3)
            for col, (lbl, sample) in zip(cols, EXAMPLES[row_start:row_start + 3]):
                if col.button(lbl, use_container_width=True, key=f"ex_{row_start}_{lbl}"):
                    st.session_state["input_text"] = sample
                    st.session_state["auto_analyze"] = True

        text = st.text_area(
            "input", key="input_text", height=130,
            label_visibility="collapsed",
            placeholder="Type or paste a social media post here…",
        )
        word_count = len(text.split()) if text.strip() else 0
        st.caption(f"{len(text)} characters · {word_count} words")

        analyze = st.button("\U0001F50D Analyze Text", use_container_width=True, type="primary")

    auto = st.session_state.get("auto_analyze", False)
    if auto:
        st.session_state["auto_analyze"] = False
    if analyze or auto:
        translated_text, detected_lang = detect_and_translate(text)
        if detected_lang != "en":
            st.info(f"🌐 Detected language: **{detected_lang.upper()}** — translated to English for analysis.")

        if model_choice == TRANSFORMER_DISPLAY_NAME:
            transformer_pipe = load_transformer()
            if transformer_pipe is None:
                st.error("Transformer unavailable. Run: `pip install transformers torch`")
                st.stop()
            with st.spinner("Running BERT-based transformer model..."):
                scores_df = transformer_scores(transformer_pipe, translated_text)
            raw_pred = scores_df.iloc[0]["Emotion"]
            top_conf = float(scores_df.iloc[0]["Confidence"])
            uncertain = top_conf < CONF_THRESHOLD
            pred = "Neutral" if uncertain else raw_pred
            clean = None
        else:
            model = tfidf_models[model_choice]
            clean = preprocess_text(translated_text)
            X = tfidf.transform([clean])
            raw_pred = model.predict(X)[0]
            scores_df = prediction_scores(model, X)
            top_conf = float(scores_df.iloc[0]["Confidence"])
            known_words = int(X.nnz)
            uncertain = (top_conf < CONF_THRESHOLD) or (known_words == 0)
            pred = "Neutral" if uncertain else raw_pred

        emoji, color = EMOTION_STYLE.get(pred, ("\U0001F3AD", ACCENT))

        with right:
            st.markdown("#### Prediction")
            conf_line = f"Confidence {top_conf*100:.1f}%"
            st.markdown(
                f"<div class='result-card'><div class='result-emoji'>{emoji}</div>"
                f"<div class='result-label' style='color:{color}'>{pred}</div>"
                f"<div class='result-conf'>{conf_line}</div></div>",
                unsafe_allow_html=True,
            )
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=top_conf * 100,
                number={"suffix": "%", "font": {"size": 26}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "borderwidth": 0,
                },
            ))
            st.plotly_chart(style_fig(gauge, height=220), use_container_width=True)

        st.markdown("#### Confidence across all emotions")
        c1, c2 = st.columns([1.3, 1])
        fig = px.bar(scores_df, x="Confidence", y="Emotion", orientation="h",
                     color="Emotion", color_discrete_map={k: v[1] for k, v in EMOTION_STYLE.items()},
                     text=scores_df["Confidence"].map(lambda v: f"{v*100:.1f}%"))
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        c1.plotly_chart(style_fig(fig), use_container_width=True)

        with c2:
            if model_choice != TRANSFORMER_DISPLAY_NAME and clean:
                st.markdown("**Words that influenced this prediction**")
                words_df = input_influential_words(clean)
                if len(words_df):
                    st.dataframe(words_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No strong keywords found. Try a longer sentence.")
                st.caption(f"Preprocessed: `{clean or '-'}`")
            else:
                st.markdown("**Model info**")
                st.info("DistilRoBERTa uses contextual embeddings — no manual feature extraction needed.")
                if detected_lang != "en":
                    st.caption(f"Original ({detected_lang.upper()}): _{text[:100]}_")
                    st.caption(f"Translated (EN): _{translated_text[:100]}_")

elif page == "Overview":
    st.markdown(
        "<div class='hero'><h1>Project Overview</h1>"
        "<p>An NLP system that classifies social media text into six emotions using classical ML models and TF-IDF / Word2Vec features.</p></div>",
        unsafe_allow_html=True,
    )
    best = results_df.sort_values("Test Accuracy", ascending=False).iloc[0]
    metric_row([
        (f"{len(data_df):,}", "Dataset samples"),
        (len(metadata["classes"]), "Emotion classes"),
        (len(results_df), "Models compared"),
        (f"{best['Test Accuracy']*100:.1f}%", "Best accuracy"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='card'><h4>Problem</h4>"
                    "<p>Social media posts carry emotions - joy, sadness, anger, fear, surprise or a neutral tone. "
                    "This app predicts that emotion instantly and explains why.</p></div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><h4>How to use</h4>"
                    "<p>Open <b>Text Analyzer</b>, paste a sentence, pick a model, and click Analyze. "
                    "You get the predicted emotion, confidence, and the most influential words.</p></div>",
                    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Emotions covered**")
    pills = " ".join(
        f"<span class='pill'>{EMOTION_STYLE.get(c, ('', ''))[0]} {c}</span>" for c in metadata["classes"]
    )
    st.markdown(pills, unsafe_allow_html=True)
    st.caption("Team: Laila, Wajee, Shree, Qistina")

elif page == "Data Explorer":
    st.markdown("<div class='hero'><h1>\U0001F4CA Data Explorer</h1><p>Browse the dataset and its class balance.</p></div>",
                unsafe_allow_html=True)
    metric_row([
        (f"{len(data_df):,}", "Total rows"),
        (int(data_df["text"].isna().sum()), "Missing text"),
        (f"{data_df['text'].str.split().str.len().mean():.1f}", "Avg words / text"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("**Sample rows**")
        st.dataframe(data_df[["text", "emotion", "clean_text"]].head(150),
                     use_container_width=True, hide_index=True, height=420)
    with c2:
        fig_pie = px.pie(class_dist_df, names="Emotion", values="Count", hole=0.55,
                         color="Emotion", color_discrete_map={k: v[1] for k, v in EMOTION_STYLE.items()},
                         title="Class distribution")
        st.plotly_chart(style_fig(fig_pie, height=420), use_container_width=True)

elif page == "Visualizations":
    st.markdown("<div class='hero'><h1>\U0001F4C8 Visualizations</h1>"
                "<p>Word cloud, class distribution, confusion matrix, model comparison, top words, plus text-length distribution.</p></div>",
                unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Words", "Data", "Models"])

    with t1:
        st.markdown("#### 1. Word Cloud")
        selected_class = st.selectbox("Emotion", ["All"] + metadata["classes"], key="wc")
        if selected_class == "All":
            text_blob = " ".join(data_df["clean_text"].dropna().astype(str))
        else:
            text_blob = " ".join(data_df.loc[data_df["emotion"] == selected_class, "clean_text"].dropna().astype(str))
        if text_blob.strip():
            wc = WordCloud(width=1100, height=420, background_color=None, mode="RGBA",
                           colormap="cool", max_words=120).generate(text_blob)
            fig_wc, ax = plt.subplots(figsize=(12, 5))
            fig_wc.patch.set_alpha(0.0)
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc, transparent=True)

        st.markdown("#### 5. Top 20 Words by Emotion")
        emotion = st.selectbox("Emotion", metadata["classes"], key="tw")
        top20 = top_words_df[top_words_df["Emotion"] == emotion].head(20)
        fig_top = px.bar(top20, x="TFIDF Score", y="Word", orientation="h",
                         color="TFIDF Score", color_continuous_scale="Purples",
                         title=f"Top 20 important words - {emotion}")
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_fig(fig_top, height=480), use_container_width=True)

    with t2:
        st.markdown("#### 2. Class Distribution")
        fig_dist = px.bar(class_dist_df, x="Emotion", y="Count", color="Emotion",
                          color_discrete_map={k: v[1] for k, v in EMOTION_STYLE.items()},
                          text_auto=True, title="Samples per emotion")
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig_dist), use_container_width=True)

        st.markdown("#### Extra. Text Length Distribution")
        fig_len = px.histogram(lengths_df, x="TextLength", color="emotion", nbins=30,
                               color_discrete_map={k: v[1] for k, v in EMOTION_STYLE.items()},
                               title="Text length distribution by emotion")
        st.plotly_chart(style_fig(fig_len), use_container_width=True)

    with t3:
        st.markdown("#### 3. Confusion Matrix")
        fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", color_continuous_scale="Purples",
                           title=f"Confusion matrix - {metadata['best_model_name']}")
        st.plotly_chart(style_fig(fig_cm, height=460), use_container_width=True)

        st.markdown("#### 4. Model Comparison")
        metrics = ["Test Accuracy", "Precision", "Recall", "Macro F1"]
        melted = results_df.melt(id_vars=["Model", "Feature Method"], value_vars=metrics,
                                 var_name="Metric", value_name="Score")
        fig_model = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group",
                           title="Model performance comparison")
        fig_model.update_xaxes(tickangle=-15)
        st.plotly_chart(style_fig(fig_model, height=460), use_container_width=True)

elif page == "Model Info":
    st.markdown("<div class='hero'><h1>\U0001F916 Model Info</h1><p>Pipeline, models, and training details.</p></div>",
                unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='card'><h4>NLP pipeline</h4>"
                    "<p>Lowercase, remove URLs, strip special characters/numbers, tokenize, "
                    "remove stopwords, stemming.</p></div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><h4>Feature extraction</h4>"
                    "<p>TF-IDF vectors and Word2Vec average embeddings, compared across two classifiers "
                    "(Logistic Regression and Linear SVM).</p></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Models trained and compared**")
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    best_model_name = metadata["best_model_name"]
    st.success(f"Best model: {best_model_name}")
    report_path = os.path.join(MODEL_DIR, "classification_report.txt")
    if os.path.exists(report_path):
        with st.expander("Full classification report"):
            st.text(open(report_path, encoding="utf-8").read())
    with st.expander("Training metadata (JSON)"):
        st.json(metadata)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Advanced NLP: Transformer Model")
    st.markdown(
        "<div class='card'><h4>DistilRoBERTa (BERT-based)</h4>"
        "<p>A <b>pre-trained transformer</b> fine-tuned on emotion classification. Unlike TF-IDF, "
        "it captures contextual meaning, negation, and long-range word dependencies via self-attention. "
        "Select DistilRoBERTa (Transformer) in the Text Analyzer to compare it against "
        "classical ML models.</p>"
        "<p style='color:#9AA5B1;font-size:0.85rem'>Model: "
        "<code>j-hartmann/emotion-english-distilroberta-base</code> (HuggingFace)</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Multi-Language Support")
    st.markdown(
        "<div class='card'><h4>Auto-detect and translate</h4>"
        "<p>Input text is detected using <b>langdetect</b>. If the text is not in English, "
        "it is automatically translated via Google Translate before analysis, enabling emotion "
        "detection in Malay, Arabic, French, Spanish, and more. A banner is shown whenever "
        "translation is applied.</p></div>",
        unsafe_allow_html=True,
    )
