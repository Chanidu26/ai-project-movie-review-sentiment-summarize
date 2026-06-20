from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import re
import math
import warnings
from dotenv import load_dotenv
load_dotenv()
import os
warnings.filterwarnings('ignore')

import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app)

# ── Load BERT sentiment model once at startup ────────────────────────────────
print("Loading BERT sentiment model...")
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model    = DistilBertForSequenceClassification.from_pretrained('./models/bert_sentiment').to(device)
bert_tokenizer = DistilBertTokenizer.from_pretrained('./models/bert_sentiment')
bert_model.eval()
print(f"✅ BERT ready on {device}")

# ── Gemini API client (replaces T5 generative summarizer) ────────────────────
# ⚠️ Hardcoded per request — replace with your real key, or better, swap this
#    back to os.environ.get("GEMINI_API_KEY") before sharing/committing this file.
#    NEVER paste a real key into chat, a notebook cell, or a committed file —
#    treat any key that's been pasted anywhere outside your local .env as
#    compromised and revoke/regenerate it at aistudio.google.com/apikey.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.5-flash"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
print(f"✅ Gemini client ready (model: {GEMINI_MODEL})")

# ── NLP helpers ──────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def predict_sentiment(text, max_len=128):
    inputs = bert_tokenizer(
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
        outputs = bert_model(input_ids, attention_mask=attention_mask)
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

# ═════════════════════════════════════════════════════════════════════════════
# METHOD 1 — TF-IDF Extractive Summarizer (original)
# Picks the most important existing sentences using TF-IDF scoring.
# ═════════════════════════════════════════════════════════════════════════════

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

def generate_summary_tfidf(review, sentiment=None, style='concise', n_sentences=2):
    """
    Method 1: TF-IDF extractive summarizer.
    Scores each sentence by word importance and picks the top N.
    Style controls which sentences are prioritised.
    """
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


# ═════════════════════════════════════════════════════════════════════════════
# METHOD 2 — Gemini Flash Generative Summarizer
# Uses prompt-engineered instructions (sent to the Gemini API) to steer
# generation, instead of prompt-prefixed input to a local T5 model.
# ═════════════════════════════════════════════════════════════════════════════

# Style prompts: each is a full instruction sent as the system instruction,
# so the model is steered the same way T5 was steered by its prefix —
# just expressed as natural-language instructions instead of a 4-word prefix.
STYLE_PROMPTS = {
    'concise':  (
        "You summarize movie reviews briefly and neutrally. "
        "Respond with ONLY the summary — 1-2 sentences, no preamble, no labels."
    ),
    'audience': (
        "You summarize movie reviews for a general audience in an accessible, "
        "engaging way, highlighting emotional reactions. "
        "Respond with ONLY the summary — 2-3 sentences, no preamble, no labels."
    ),
    'critic': (
        "You summarize movie reviews the way a film critic would, focusing on "
        "evaluative judgments (performances, direction, pacing, craft). "
        "Respond with ONLY the summary — 2-3 sentences, no preamble, no labels."
    ),
    'cot': (
        "Think step by step: first identify the reviewer's opening impression, "
        "then their main supporting point, then any contrasting or qualifying "
        "remark (e.g. 'but', 'however'). Use that reasoning to write a summary "
        "that reflects all three beats. "
        "Respond with ONLY the final summary — 2-4 sentences, no preamble, "
        "no labels, and do NOT show your step-by-step reasoning in the output."
    ),
}

# Generation parameters per style — max_output_tokens replaces T5's
# max_length; Gemini doesn't take num_beams/length_penalty, so style
# strength instead comes from the instruction text above plus temperature.
STYLE_PARAMS = {
    'concise':  {'max_output_tokens': 150,  'temperature': 0.3},
    'audience': {'max_output_tokens': 250, 'temperature': 0.5},
    'critic':   {'max_output_tokens': 280, 'temperature': 0.5},
    'cot':      {'max_output_tokens': 350, 'temperature': 0.4},
}

def generate_summary_t5(review, sentiment_label=None, style='concise'):
    """
    Method 2: Gemini Flash generative summarizer (formerly T5).
    Sends a style-specific system instruction + the review to the Gemini
    API. The model generates a brand-new summary — not extracted sentences.
    sentiment_label is optionally passed as extra context, same role it
    played as the bracketed hint fed into T5's input text.

    NOTE: function name kept as generate_summary_t5 so call sites
    (e.g. /analyse route) don't need to change.
    """
    text              = clean_text(review)
    system_instruction = STYLE_PROMPTS.get(style, STYLE_PROMPTS['concise'])
    params            = STYLE_PARAMS.get(style, STYLE_PARAMS['concise'])

    user_content = f"[Detected sentiment: {sentiment_label}]\n\n{text}" if sentiment_label else text

    response = gemini_client.models.generate_content(
    model=GEMINI_MODEL,
    contents=user_content,
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=params['max_output_tokens'],
        temperature=params['temperature'],
        thinking_config=types.ThinkingConfig(thinking_budget=0),  # turn off thinking
      ),
    )

    return response.text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# API routes
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/analyse', methods=['POST'])
def analyse():
    data   = request.get_json(silent=True) or {}
    review = data.get('review', '').strip()
    # method: 'tfidf' (default) or 'gemini'
    method = data.get('method', 'tfidf').lower()

    # 🔍 DEBUG — remove once everything works. Shows exactly what the
    # frontend is sending, which is the #1 cause of unexpected 400s.
    print(f"[/analyse] received method={method!r} review_len={len(review)}")

    if not review:
        return jsonify({'error': 'No review provided'}), 400
    if method not in ('tfidf', 'gemini'):
        return jsonify({'error': f"method must be 'tfidf' or 'gemini' (got {method!r})"}), 400

    sentiment = predict_sentiment(review)
    label     = sentiment['label'].lower()

    if method == 'gemini':
        try:
            summaries = {
                style: generate_summary_t5(review, label, style=style)
                for style in ['concise', 'audience', 'critic', 'cot']
            }
        except Exception as e:
            # Surfaces real Gemini errors (auth, rate limit, network) instead
            # of a silent failure or an unrelated-looking 500/400.
            print(f"[/analyse] Gemini call failed: {e}")
            return jsonify({'error': f'Gemini generation failed: {str(e)}'}), 502
    else:
        summaries = {
            style: generate_summary_tfidf(review, label, style=style)
            for style in ['concise', 'audience', 'critic', 'cot']
        }

    return jsonify({
        'sentiment': sentiment,
        'summaries': summaries,
        'word_count': len(review.split()),
        'method': method
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device)})


if __name__ == '__main__':
    app.run(port=5000, debug=False)