from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import re
import math
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

app = Flask(__name__)
CORS(app)  # allow the HTML file to call this API

# ── Load model once at startup ───────────────────────────────────────────────
print("Loading BERT model...")
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = DistilBertForSequenceClassification.from_pretrained('./models/bert_sentiment').to(device)
tokenizer = DistilBertTokenizer.from_pretrained('./models/bert_sentiment')
model.eval()
print(f"✅ Model ready on {device}")

# ── NLP helpers ──────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def predict_sentiment(text, max_len=128):
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids      = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    label = 'Positive' if probs[1] > 0.5 else 'Negative'
    return {
        'label':         label,
        'positive_prob': round(float(probs[1]), 4),
        'negative_prob': round(float(probs[0]), 4)
    }

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sentence_tfidf_scores(sentences):
    word_doc_freq   = Counter()
    sent_word_lists = []
    for sent in sentences:
        words = [w.lower() for w in word_tokenize(sent)
                 if w.isalpha() and w.lower() not in stop_words]
        sent_word_lists.append(words)
        word_doc_freq.update(set(words))
    N = len(sentences)
    scores = []
    for words in sent_word_lists:
        if not words:
            scores.append(0)
            continue
        tf    = Counter(words)
        score = sum((tf[w] / len(words)) * math.log(N / (word_doc_freq[w] + 1)) for w in tf)
        scores.append(score / len(words))
    return scores

def generate_summary(review, sentiment=None, style='concise', n_sentences=2):
    text      = clean_text(review)
    sentences = sent_tokenize(text)
    if len(sentences) <= 2:
        return text

    base_scores = sentence_tfidf_scores(sentences)

    if style == 'concise':
        ranked  = sorted(zip(base_scores, range(len(sentences))), reverse=True)
        top_idx = sorted([idx for _, idx in ranked[:n_sentences]])
        return ' '.join(sentences[i] for i in top_idx)

    elif style == 'audience':
        emotion_words = {'love','loved','hate','hated','amazing','terrible',
                         'beautiful','boring','exciting','funny','awful','perfect',
                         'incredible','fantastic','brilliant','worst','best'}
        adjusted = []
        for score, sent in zip(base_scores, sentences):
            words          = set(sent.lower().split())
            has_emote      = bool(words & emotion_words)
            length         = len(sent.split())
            length_penalty = max(0.5, 1 - (length - 15) * 0.02) if length > 15 else 1.0
            adjusted.append((score * 0.4 + (0.6 if has_emote else 0)) * length_penalty)
        ranked  = sorted(zip(adjusted, range(len(sentences))), reverse=True)
        top_idx = sorted([idx for _, idx in ranked[:n_sentences]])
        concise_idx = sorted([idx for _, idx in
                               sorted(zip(base_scores, range(len(sentences))),
                                      reverse=True)[:n_sentences]])
        if top_idx == concise_idx and len(sentences) > n_sentences + 1:
            all_ranked = sorted(zip(adjusted, range(len(sentences))), reverse=True)
            top_idx    = sorted([idx for _, idx in all_ranked[:n_sentences+1]])[:n_sentences]
        return ' '.join(sentences[i] for i in top_idx)

    elif style == 'critic':
        judgment = {'brilliant','excellent','outstanding','masterpiece',
                    'poor','weak','disappointing','flawed','waste',
                    'great','good','bad','worst','best','average',
                    'rushed','unanswered','must-watch','incredible'}
        adjusted = [score + (0.6 if set(sent.lower().split()) & judgment else 0)
                    for score, sent in zip(base_scores, sentences)]
        ranked  = sorted(zip(adjusted, range(len(sentences))), reverse=True)
        top_idx = sorted([idx for _, idx in ranked[:n_sentences]])
        return ' '.join(sentences[i] for i in top_idx)

    elif style == 'cot':
        step1 = sentences[0]
        mid   = list(enumerate(base_scores[1:-1], 1))
        step2 = sentences[max(mid, key=lambda x: x[1])[0]] if mid else sentences[0]
        qualifiers = ['but','however','although','despite','yet','unfortunately']
        step3 = next((s for s in sentences
                      if any(q in s.lower() for q in qualifiers)), sentences[-1])
        parts = [step1]
        if step2 != step1:     parts.append(step2)
        if step3 not in parts: parts.append(step3)
        return ' '.join(parts[:3])

    return sentences[0]

# ── API route ────────────────────────────────────────────────────────────────
@app.route('/analyse', methods=['POST'])
def analyse():
    data   = request.get_json()
    review = data.get('review', '').strip()

    if not review:
        return jsonify({'error': 'No review provided'}), 400

    sentiment  = predict_sentiment(review)
    label      = sentiment['label'].lower()
    summaries  = {
        style: generate_summary(review, label, style=style)
        for style in ['concise', 'audience', 'critic', 'cot']
    }

    return jsonify({
        'sentiment': sentiment,
        'summaries': summaries,
        'word_count': len(review.split())
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device)})

if __name__ == '__main__':
    app.run(port=5000, debug=False)
