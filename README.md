# Movie Review Summarizer — Advanced AI Final Project

An end-to-end project for sentiment analysis and summarization of movie reviews. It includes data preprocessing, word-embedding experiments, a fine-tuned DistilBERT sentiment classifier, extractive TF‑IDF summarization, and a generative summarizer integration (Gemini API).

## Quick summary
- Sentiment: DistilBERT model stored under `models/bert_sentiment`
- Extractive summarization: TF‑IDF sentence ranking (in `app.py`)
- Generative summarization: Gemini API wrapper (in `app.py`)
- Notebooks: exploratory analysis, model training, prompt engineering, evaluation

## Techniques We Used
- **Natural Language Processing (NLP):** text cleaning, tokenization, sentence splitting, stopword removal; implemented in `app.py` and preprocessing notebook (`01_data_exploration_preprocessing.ipynb`).
- **Word Embeddings:** Word2Vec model (`models/word2vec_imdb.model`) trained in Notebook 1 (100‑dim vectors used for analysis/visualization).
- **Transformer-based Models / LLMs:** DistilBERT fine-tuned for binary sentiment classification (loaded from `models/bert_sentiment`); training code and experiments in `02_bert_sentiment_classification.ipynb`.
- **Generative AI:** originally experimented with a T5-style approach in Notebook 3; production/generative summarization is implemented via the Gemini API integration in `app.py` (requires `GEMINI_API_KEY`).
- **Model Training Approaches:** transfer learning (DistilBERT fine-tuning) and standard supervised training for baselines (Logistic Regression, SVM) covered in the notebooks.
- **Prompt Engineering:** systematic prompt templates and style-specific instructions used for the generative summarizer (see `app.py` STYLE_PROMPTS and `03_t5_summarization_prompt_engineering.ipynb`).
- **Extractive Summarization:** TF‑IDF sentence scoring and style-aware selection implemented in `app.py` (`generate_summary_tfidf`).

Notes: items not used in this project — autoencoders, GANs, and diffusion models are outside this repo's scope (listed here for completeness of techniques in the course context).

## Repository layout
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

## Getting started

### Create and activate a Python virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### Install required packages (suggested):

```bash
pip install pandas numpy matplotlib seaborn nltk gensim wordcloud scikit-learn
pip install transformers torch tqdm rouge-score flask flask-cors python-dotenv google-genai
```

### Download NLTK data (once):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Running the Notebooks

The four notebooks must be run **in order** (01 → 02 → 03 → 04). Each one depends on outputs from the previous step.

#### Notebook 1: Data Exploration & Preprocessing (`01_data_exploration_preprocessing.ipynb`)
- **What it does:** Loads `data/IMDB_Dataset.csv`, performs EDA, cleans text, balances classes, and splits into train/val/test sets.
- **Prerequisites:** `data/IMDB_Dataset.csv` must exist.
- **Outputs generated:**
  - `data/train.csv` (training split, ~40K reviews)
  - `data/val.csv` (validation split, ~5K reviews)
  - `data/test.csv` (test split, ~5K reviews)
  - `models/word2vec_imdb.model` (Word2Vec embeddings, 100-dim)
  - `outputs/01_eda_distribution.png` (class distribution chart)
  - `outputs/02_wordclouds.png` (word clouds for positive/negative)
  - `outputs/03_word2vec_tsne.png` (t-SNE visualization of embeddings)
- **Run time:** ~10–20 minutes (depending on CSV size and machine).

#### Notebook 2: BERT Sentiment Classification (`02_bert_sentiment_classification.ipynb`)
- **What it does:** Fine-tunes DistilBERT on the sentiment classification task. Trains baselines (Naive Bayes, Logistic Regression, SVM) for comparison.
- **Prerequisites:** `data/train.csv` and `data/val.csv` from Notebook 1.
- **Outputs generated:**
  - `models/bert_sentiment/` (fine-tuned DistilBERT weights, config, tokenizer)
  - `outputs/04_bert_training_curves.png` (loss and accuracy curves)
  - `outputs/09_bert_confusion_matrix.png` (confusion matrix for test set)
  - Baseline model accuracies printed (Naive Bayes ~85%, Logistic Regression ~88%, SVM ~89%, DistilBERT ~93%).
- **Run time:** ~2–3 hours on CPU; ~20–30 minutes on GPU.
- **GPU note:** If GPU is available, PyTorch will use it automatically. Check with `torch.cuda.is_available()` in a cell.

#### Notebook 3: T5 Summarization & Prompt Engineering (`03_t5_summarization_prompt_engineering.ipynb`)
- **What it does:** Experiments with extractive (TF‑IDF) and generative summarization (T5 or API-based). Tests style-specific prompts (concise, audience, critic, chain-of-thought).
- **Prerequisites:** `data/test.csv` from Notebook 1; optionally uses sentiment predictions from Notebook 2.
- **Outputs generated:**
  - `outputs/06_prompt_rouge_evaluation.png` (ROUGE scores by style)
  - `outputs/08_keywords.png` (keyword extraction visualization)
  - Example summaries and ROUGE metrics printed.
- **Run time:** ~30 minutes to 1 hour (T5 generation or API calls).

#### Notebook 4: Evaluation & Model Comparison (`04_evaluation_baseline_comparison.ipynb`)
- **What it does:** Comprehensive evaluation: confusion matrices, ROUGE scores, per-class metrics, and comparison charts.
- **Prerequisites:** Outputs from Notebooks 1–3.
- **Outputs generated:**
  - `outputs/07_model_comparison.png` (bar chart comparing model accuracies)
  - `outputs/rouge_scores_by_prompt.csv` (detailed ROUGE results)
  - Performance tables and error analysis printed.
- **Run time:** ~5–10 minutes.

#### After All Notebooks
Once all four notebooks are complete:
- ✅ `data/train.csv`, `data/val.csv`, `data/test.csv` exist in `data/`
- ✅ `models/bert_sentiment/` is ready for sentiment inference
- ✅ `models/word2vec_imdb.model` is saved
- ✅ All output charts are in `outputs/`
- ✅ You can now run the Flask API (see below)


## Running the Web API

- Configure environment: create a `.env` file or set `GEMINI_API_KEY` if you plan to use the Gemini generative summarizer. If you don't have a Gemini key, the default TF‑IDF summarizer still works.
- Start the server:

```bash
python app.py
```

The server listens on port 5000 by default.

### Using the Web Frontend (`index.html`)

Once the Flask server is running:
1. Open your browser and navigate to `http://localhost:5000/`
2. You should see the **Movie Review Analyser** interface (styled dark theme with accent colors)
3. Enter or paste a movie review in the text area
4. Choose a summarization method:
   - **TF‑IDF** (default, fast, no API key needed)
   - **Gemini** (requires `GEMINI_API_KEY` in `.env`, more fluent summaries)
5. Click **Analyse** to process the review
6. Results display:
   - **Sentiment** (positive/negative probability)
   - **Summary** variations (concise, audience, critic, chain-of-thought)
   - **Word count** of the original review

The frontend is a simple, responsive single-page application that calls the `/analyse` endpoint.

### API Endpoints

- **GET `/health`** — returns basic server/device info (CPU or GPU).
- **POST `/analyse`** — sends a review for analysis.
  - Request body: `{"review": "<your review text>", "method": "tfidf"}` or `{"method":"gemini"}`
  - Response: JSON with sentiment probabilities, summaries (4 styles), and word count.

Example (curl):

```bash
curl -X POST http://localhost:5000/analyse \
	-H "Content-Type: application/json" \
	-d '{"review":"I loved the movie, it was brilliant and moving.", "method":"tfidf"}'
```
