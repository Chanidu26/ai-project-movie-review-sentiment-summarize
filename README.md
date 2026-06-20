# Movie Review Summarizer — Advanced AI Final Project

An end-to-end project for sentiment analysis and summarization of movie reviews. It includes data preprocessing, word-embedding experiments, a fine-tuned DistilBERT sentiment classifier, extractive TF‑IDF summarization, and a generative summarizer integration (Gemini API).

**Quick summary:**
- Sentiment: DistilBERT model stored under `models/bert_sentiment`
- Extractive summarization: TF‑IDF sentence ranking (in `app.py`)
- Generative summarization: Gemini API wrapper (in `app.py`)
- Notebooks: exploratory analysis, model training, prompt engineering, evaluation

**Techniques We Used:**
- **Natural Language Processing (NLP):** text cleaning, tokenization, sentence splitting, stopword removal; implemented in `app.py` and preprocessing notebook (`01_data_exploration_preprocessing.ipynb`).
- **Word Embeddings:** Word2Vec model (`models/word2vec_imdb.model`) trained in Notebook 1 (100‑dim vectors used for analysis/visualization).
- **Transformer-based Models / LLMs:** DistilBERT fine-tuned for binary sentiment classification (loaded from `models/bert_sentiment`); training code and experiments in `02_bert_sentiment_classification.ipynb`.
- **Generative AI:** originally experimented with a T5-style approach in Notebook 3; production/generative summarization is implemented via the Gemini API integration in `app.py` (requires `GEMINI_API_KEY`).
- **Model Training Approaches:** transfer learning (DistilBERT fine-tuning) and standard supervised training for baselines (Logistic Regression, SVM) covered in the notebooks.
- **Prompt Engineering:** systematic prompt templates and style-specific instructions used for the generative summarizer (see `app.py` STYLE_PROMPTS and `03_t5_summarization_prompt_engineering.ipynb`).
- **Extractive Summarization:** TF‑IDF sentence scoring and style-aware selection implemented in `app.py` (`generate_summary_tfidf`).

Notes: items not used in this project — autoencoders, GANs, and diffusion models are outside this repo's scope (listed here for completeness of techniques in the course context).

**Repository layout**
```
movie_review_project/
├── app.py                      # Flask API server (analyse, health)
├── index.html                  # Simple frontend for manual testing
├── data/                       # IMDB_Dataset.csv and train/val/test splits
├── models/
│   ├── bert_sentiment/         # Hugging Face-style DistilBERT files
│   └── word2vec_imdb.model     # Word2Vec model from Notebooks
├── notebooks/                  # Analysis, training and evaluation notebooks
├── outputs/                    # Generated figures and evaluation CSVs
└── README.md
```

**Getting started:**

1) (Optional) create and activate a Python virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2) Install required packages (suggested):

```bash
pip install pandas numpy matplotlib seaborn nltk gensim wordcloud scikit-learn
pip install transformers torch tqdm rouge-score flask flask-cors python-dotenv google-genai
```

3) Download NLTK data (once):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

Running the web API

- Configure environment: create a `.env` file or set `GEMINI_API_KEY` if you plan to use the Gemini generative summarizer. If you don't have a Gemini key, the default TF‑IDF summarizer still works.
- Start the server:

```bash
python app.py
```

The server listens on port 5000 by default.

API (HTTP)
- GET `/health` — returns basic server/device info.
- POST `/analyse` — JSON body: `{"review": "<text>", "method": "tfidf"}` or `{"method":"gemini"}` to use the generative route.

Example (curl):

```bash
curl -X POST http://localhost:5000/analyse \
	-H "Content-Type: application/json" \
	-d '{"review":"I loved the movie, it was brilliant and moving.", "method":"tfidf"}'
```

**models & data:**
- Place the original dataset `IMDB_Dataset.csv` in the `data/` folder if you want to re-run preprocessing and splits.
- The DistilBERT sentiment model is expected at `models/bert_sentiment` (Hugging Face format). `app.py` loads this at startup.
- The Gemini (generative) path uses `google.genai` and requires a valid `GEMINI_API_KEY` in the environment.

**Notebooks**
- `01_data_exploration_preprocessing.ipynb`: EDA and preprocessing, creates `train.csv`, `val.csv`, `test.csv`.
- `02_bert_sentiment_classification.ipynb`: Fine-tune DistilBERT for sentiment.
- `03_t5_summarization_prompt_engineering.ipynb`: Prompt experiments and generative summarization.
- `04_evaluation_baseline_comparison.ipynb`: Metrics and model comparisons (ROUGE, confusion matrices, charts saved in `outputs/`).

**Outputs:**
- Figures and evaluation CSVs are in `outputs/` (training curves, confusion matrix, ROUGE scores, etc.).
