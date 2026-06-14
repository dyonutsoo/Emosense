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
The trained model files are already included in `models/`, so you can skip
`python train_models.py` unless you want to retrain. You only need
`pip install -r requirements.txt` and `streamlit run app.py`.

## Running on Windows (recommended)

Use **Python 3.12**, not the newest release. See troubleshooting below for why.

1. Check which Python versions you have:
   ```powershell
   py list
   ```
   If you don't see a `3.12` entry, install it:
   ```powershell
   py install 3.12
   ```

2. From the project folder, create a virtual environment with 3.12 and install:
   ```powershell
   cd path\to\Emosense
   py -3.12 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Run the app:
   ```powershell
   .venv\Scripts\python.exe -m streamlit run app.py
   ```
   On first run Streamlit asks for an email — leave it blank and press Enter.
   Then open http://localhost:8501 in your browser (it may not open by itself).

To start the app again later, you only need step 3 (no reinstalling).

## Troubleshooting

**`pip` / `python` / `streamlit` is not recognized**
Python isn't installed or isn't on your PATH. Install Python 3.12 (see above)
and always call it through the virtual environment:
`.venv\Scripts\python.exe -m <command>`.

**"Python was not found; run without arguments to install from the Microsoft Store"**
This is a Windows placeholder, not a real Python. Install Python 3.12 with
`py install 3.12`.

**gensim fails to build: "Microsoft Visual C++ 14.0 or greater is required"**
This happens when you use a Python version too new for gensim (e.g. 3.14).
pip then tries to compile gensim from source, which needs a C++ compiler.
Do **not** install the C++ Build Tools. Instead use Python 3.12, which has a
prebuilt gensim wheel — no compiler needed. Recreate the venv with
`py -3.12 -m venv .venv` and reinstall.

**Installs still use the wrong Python version**
On Windows, `.venv\Scripts\activate` can silently fail to activate (PowerShell
script policy). Either run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and then activate,
or just skip activation and call the venv Python directly:
`.venv\Scripts\python.exe -m ...`. Confirm with:
```powershell
.venv\Scripts\python.exe --version   # should say 3.12.x
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
