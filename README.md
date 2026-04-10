### Fake News Detection in Philippine News using Support Vector Machine (SVM)

> **Developed by:** Team LiveANet  
> **Project Type:** Undergraduate Thesis / Research Prototype  
> **Domain:** Natural Language Processing · Machine Learning · Browser Extension

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model Training](#model-training)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [TF-IDF Vectorization](#tf-idf-vectorization)
  - [SVM Model & Hyperparameter Tuning](#svm-model--hyperparameter-tuning)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Saving the Model](#saving-the-model)
- [Backend Server](#backend-server)
  - [API Endpoint](#api-endpoint)
  - [Text Summarization](#text-summarization)
- [Chrome Extension (Prototype)](#chrome-extension-prototype)
  - [How It Works](#how-it-works)
  - [Extension Files](#extension-files)
  - [Installing the Extension](#installing-the-extension)
- [Running the Project](#running-the-project)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Start the Backend Server](#start-the-backend-server)
  - [Load the Extension](#load-the-extension)
- [Usage](#usage)
- [Credibility Thresholds](#credibility-thresholds)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

We developed a fake news detection system designed specifically for Philippine online news articles. It combines a trained **Support Vector Machine (SVM)** classifier with a **Google Chrome browser extension**, enabling users to analyze the credibility of any news article they are reading in real time.

The system:
- Extracts the text content of the active browser tab
- Sends it to a local Flask API server
- Runs it through a trained SVM model with TF-IDF features
- Returns a **credibility verdict** (Credible / Suspicious), a **probability score**, and a **text summary**
- Displays results directly in the extension popup

---

## Project Structure

```
Fake-News-in-Philippine-News-using-SVM-Algorithm-Training-and-Prototype/
│
├── SVM/
│   └── fake_news_detector/
│       ├── Train_SVM.ipynb              # Jupyter notebook for model training
│       ├── svm_model.pkl                # Saved trained SVM model (CalibratedClassifierCV)
│       └── tfidf_vectorizer.pkl         # Saved TF-IDF vectorizer
│
├── prototype_svm_server.py              # Flask backend API server
│
└── chrome_extension/
    ├── manifest.json                    # Chrome extension manifest (MV3)
    ├── popup.html                       # Extension popup UI
    ├── popup.js                         # Popup logic and API communication
    ├── popup.css                        # Popup styles
    ├── content.js                       # Content script for page text extraction & highlighting
    └── img/
        ├── icon16.png
        ├── icon48.png
        └── icon128.png
```

---

## System Architecture

```
┌─────────────────────────┐        ┌──────────────────────────────┐
│   Chrome Browser        │        │   Flask Backend (localhost)   │
│                         │        │                               │
│  ┌───────────────────┐  │  HTTP  │  ┌────────────────────────┐  │
│  │  Extension Popup  │──┼──POST──┼─▶│  /predict endpoint     │  │
│  │  (popup.js)       │  │        │  │                        │  │
│  └────────┬──────────┘  │        │  │  1. Preprocess text    │  │
│           │ chrome msg  │        │  │  2. TF-IDF transform   │  │
│  ┌────────▼──────────┐  │        │  │  3. SVM prediction     │  │
│  │  Content Script   │  │        │  │  4. LSA summarization  │  │
│  │  (content.js)     │  │        │  └────────────────────────┘  │
│  │  - Extract text   │  │        │                               │
│  │  - Highlight words│  │        │  svm_model.pkl               │
│  └───────────────────┘  │        │  tfidf_vectorizer.pkl         │
└─────────────────────────┘        └──────────────────────────────┘
```

---

## Dataset

The model is trained on a custom dataset hosted on Hugging Face:

**Dataset:** [`nixbel/dataset_train_thesis`](https://huggingface.co/datasets/nixbel/dataset_train_thesis)

| Column     | Description                                      |
|------------|--------------------------------------------------|
| `Headline` | Title of the news article                        |
| `Authors`  | Author(s) of the article                         |
| `Date`     | Publication date and time                        |
| `Content`  | Full body text of the article                    |
| `Brand`    | News outlet (e.g., Rappler, and other PH sources)|
| `URL`      | Source URL of the article                        |
| `Label`    | `0` = Credible, `1` = Suspicious                 |

The dataset was collected from Philippine news outlets and labeled to distinguish credible news from suspicious/fake news articles.

---

## Model Training

The full training pipeline is found in `Train_SVM.ipynb`.

### Preprocessing Pipeline

Each article's `Content` field is passed through the following steps:

1. **Lowercase conversion** — Normalizes text casing
2. **Special character removal** — Strips punctuation, numbers, and symbols using regex `[^a-zA-Z\s]`
3. **Tokenization** — Splits text into individual tokens using NLTK's `word_tokenize`
4. **Stopword removal** — Filters out common English stopwords using NLTK's stopword corpus
5. **Lemmatization** *(server-side)* — Reduces words to their base/root form using `WordNetLemmatizer`

```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)
```

### TF-IDF Vectorization

Text features are extracted using **Term Frequency–Inverse Document Frequency (TF-IDF)**:

| Parameter       | Value         |
|-----------------|---------------|
| `max_features`  | 5,000         |
| `ngram_range`   | (1, 2) — unigrams and bigrams |

```python
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

### SVM Model & Hyperparameter Tuning

The model uses a **Support Vector Classifier (SVC)** with an **RBF (Radial Basis Function) kernel**. Probability estimates are enabled via Platt scaling (`probability=True`).

Hyperparameters are optimized with **GridSearchCV**:

| Hyperparameter | Values Searched      |
|----------------|----------------------|
| `C`            | `[0.1, 1, 10, 100]`  |
| `gamma`        | `['scale', 'auto', 0.1, 0.01]` |

```python
svm = SVC(kernel='rbf', probability=True, random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01]
}
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
```

The best estimator from `GridSearchCV` is wrapped in **`CalibratedClassifierCV`** to produce reliable probability outputs, then saved as `svm_model.pkl`.

### Evaluation Metrics

The trained model is evaluated using:

- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score per class)
- **Confusion Matrix** (visualized with Seaborn heatmap)
- **ROC Curve & AUC Score**
- **Precision-Recall Curve & AUC**

### Saving the Model

Both the model and vectorizer are serialized with `pickle` for use by the Flask server:

```python
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(best_svm, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
```

---

## Backend Server

**File:** `prototype_svm_server.py`  
**Framework:** Flask  
**Port:** `5000` (default)

The server loads both `.pkl` files on startup and exposes a single REST API endpoint.

### API Endpoint

#### `POST /predict`

**Request Body:**
```json
{
  "text": "Full article text content goes here..."
}
```

**Response Body:**
```json
{
  "credibility": "Credible",
  "suspicious_probability": 0.1342,
  "summary": "Three-sentence LSA summary of the article..."
}
```

| Response Field          | Type    | Description                                           |
|-------------------------|---------|-------------------------------------------------------|
| `credibility`           | string  | `"Credible"` or `"Suspicious"`                        |
| `suspicious_probability`| float   | Probability (0.0–1.0) of the article being suspicious |
| `summary`               | string  | Auto-generated 3-sentence article summary             |

### Text Summarization

The server uses **LSA (Latent Semantic Analysis)** summarization via the `sumy` library to generate a 3-sentence summary of each article:

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def generate_summary(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])
```

If summarization fails, it falls back to the first 200 characters of the article.

---

## Chrome Extension (Prototype)

### How It Works

1. The user navigates to any news article in Chrome.
2. The user clicks the **E For Real** extension icon in the toolbar.
3. The popup appears and the user clicks **"Analyze Content"**.
4. The extension's **content script** (`content.js`) extracts the article text from the page (prioritizing `<article>` and `<main>` tags, falling back to `<body>`).
5. The extracted text is sent via `fetch` to the local Flask server at `http://localhost:5000/predict`.
6. The server returns prediction results, which are displayed in the popup:
   - Credibility verdict (color-coded green/red)
   - Suspicious probability percentage
   - Animated progress bar
   - Article summary with color-coded highlight
7. The content script can also **highlight influential words** directly on the page in green (credible) or red (suspicious).

### Extension Files

| File            | Purpose                                                          |
|-----------------|------------------------------------------------------------------|
| `manifest.json` | Extension configuration — MV3, permissions, content scripts     |
| `popup.html`    | The UI rendered when the extension icon is clicked              |
| `popup.js`      | Handles button click, messaging, API call, and result rendering |
| `popup.css`     | Styling for the popup UI                                         |
| `content.js`    | Injected into web pages — extracts text and highlights words     |

### Installing the Extension

> The extension is an **unpacked Chrome extension** and must be loaded manually in developer mode.

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **"Load unpacked"**
4. Select the `chrome_extension/` folder (the folder containing `manifest.json`)
5. The **E For Real** extension will appear in your extensions bar

> **Note:** Ensure the icon files exist at `img/icon16.png`, `img/icon48.png`, and `img/icon128.png` relative to `manifest.json`.

---

## Running the Project

### Prerequisites

- Python 3.8+
- Google Chrome browser
- `pip` package manager

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/Fake-News-in-Philippine-News-using-SVM.git
cd Fake-News-in-Philippine-News-using-SVM
pip install flask flask-cors joblib nltk scikit-learn sumy
```

Download required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Start the Backend Server

Make sure `svm_model.pkl` and `tfidf_vectorizer.pkl` are placed in the correct path referenced in `prototype_svm_server.py`. Update the paths if needed:

```python
# In prototype_svm_server.py — update these paths to match your setup:
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
calibrated_model = joblib.load('svm_model.pkl')
```

Then start the server:

```bash
python prototype_svm_server.py
```

The server will run at: `http://localhost:5000`

### Load the Extension

Follow the [Installing the Extension](#installing-the-extension) steps above to load the unpacked extension in Chrome.

---

## Usage

1. Start the Flask backend server (`python prototype_svm_server.py`)
2. Open Chrome and navigate to any Philippine news article
3. Click the **E For Real** extension icon
4. Click **"Analyze Content"**
5. View the results:
   - **Green label** = Credible article
   - **Red label** = Suspicious article
   - **Progress bar** = Visual indicator of suspicious probability
   - **Summary panel** = Key sentences from the article, color-coded by verdict

---

## Credibility Thresholds

| Suspicious Probability | Verdict      |
|------------------------|--------------|
| < 0.80 (< 80%)         | ✅ Credible   |
| ≥ 0.80 (≥ 80%)         | ⚠️ Suspicious |

The threshold of **0.80** was chosen to minimize false positives — only articles with a high probability of being suspicious are flagged, reducing the risk of incorrectly labeling legitimate news.

---

## Limitations

- **English-only preprocessing** — Stopwords and tokenization are configured for English. Filipino/Tagalog articles may not be preprocessed optimally.
- **Local server dependency** — The extension requires a running local Flask server. It will not work without it.
- **Dataset scope** — The training dataset is focused on Philippine news sources; performance on international news outlets may vary.
- **Static threshold** — The 0.80 credibility threshold is fixed and not dynamically adjustable from the UI.
- **No persistent history** — The extension does not store past analysis results across sessions.
- **Word highlighting not fully wired** — The `highlightWords` action in `content.js` is implemented but the current `popup.js` does not send credible/suspicious word lists to the content script after analysis.

---

## Future Improvements

- [ ] Add support for Filipino/Tagalog language preprocessing
- [ ] Train on a larger and more diverse Philippine news dataset
- [ ] Integrate word importance extraction from TF-IDF to power the word highlighting feature
- [ ] Deploy the backend to a cloud server so the extension works without a local setup
- [ ] Add adjustable credibility threshold in the extension settings
- [ ] Implement result history and per-domain statistics
- [ ] Explore deep learning alternatives (e.g., fine-tuned BERT for Filipino news)
- [ ] Add a feedback mechanism for users to report incorrect predictions

---

## License

This project was developed as part of an academic thesis by **Team LiveANet**. All rights reserved. For educational and research use only.

---

## Author

Jacques Nico Belmonte - Programmer | John Louie Abenir - Project Leader | Kenn Louise Comprado - Technical Writer
