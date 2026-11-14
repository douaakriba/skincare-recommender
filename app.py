"""
Skincare search + recommendation system  (Exercise 2: Query Expansion)

Pipeline implemented for every search:
  1. Identify query terms           -> identify_terms()
  2. Expand query terms             -> expand_query()
  3. Methods of query expansion     -> synonym / related-term / concept /
                                       feedback-based / automatic (pseudo-relevance feedback)
  4. Re-rank results                -> BM25 or TF-IDF first stage, then hybrid re-ranking
Recommendation: content-based (TF-IDF profile + skin-type / concern match + quality prior).
"""
import os, re, math, threading
from collections import Counter, defaultdict
from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

app = Flask(__name__)
DATASET_ID = "Yaminii/skincare1"
DATA_PATH = os.getenv("SKINCARE_DATA_PATH", "data/skincare1.csv")
MAX_ROWS = int(os.getenv("MAX_ROWS", "0"))  # 0 = full dataset

# ----------------------------------------------------------------------------
# Domain knowledge used by the query-expansion methods (transparent rules)
# ----------------------------------------------------------------------------
SYNONYMS = {
    "pimples": ["acne", "breakouts", "blemishes"], "pimple": ["acne", "breakouts"],
    "spots": ["dark spots", "hyperpigmentation", "pigmentation"],
    "wrinkles": ["fine lines", "aging", "anti-aging"],
    "oily": ["excess sebum", "oil control", "shine"],
    "dry": ["dehydrated", "dryness", "moisture"],
    "blackheads": ["whitehead/blackhead", "pores", "comedones"],
    "blackhead": ["blackheads", "pores", "comedones"],
    "glow": ["brightening", "radiance", "dullness"],
    "redness": ["irritation", "sensitive", "skin soothing"],
    "sunscreen": ["sun protection", "spf", "uv protection"],
    "moisturizer": ["moisturiser", "hydration", "moisturizing cream"],
    "cleanser": ["face wash", "cleansing", "face cleanser"],
}
RELATED = {
    "acne": ["salicylic acid", "niacinamide", "pores", "oil control"],
    "dryness": ["hyaluronic acid", "glycerin", "ceramides", "hydration"],
    "hydration": ["hyaluronic acid", "glycerin", "moisturizer", "barrier repair"],
    "dark spots": ["vitamin c", "niacinamide", "pigmentation", "brightening"],
    "sun protection": ["sunscreen", "spf", "uv protection"],
    "pores": ["oil control", "blackheads", "exfoliation"],
    "sensitive": ["skin soothing", "centella", "irritation", "barrier repair"],
}
CONCEPTS = {
    "acne": ["blemish care", "breakout care", "oil regulation"],
    "hydration": ["moisture retention", "skin barrier support"],
    "pigmentation": ["uneven skin tone", "dark spot care"],
    "exfoliation": ["dead skin removal", "skin texture"],
    "sun protection": ["uv protection", "photoprotection"],
}
# Words a quiz chip / filter may use  ->  words that really appear in product data
CONCERN_ALIASES = {
    "glow": ["glow", "brighten", "radiance", "dullness", "dull"],
    "hydration": ["hydrat", "dryness", "dehydrat", "moistur"],
    "acne": ["acne", "blemish", "breakout", "pimple"],
    "dark spots": ["dark spot", "pigmentation", "hyperpigmentation", "discolor"],
    "pigmentation": ["pigmentation", "dark spot", "uneven", "discolor"],
    "pores": ["pore", "blackhead", "whitehead", "sebum", "oil"],
    "exfoliation": ["exfoliat", "dead skin", "texture", "peel"],
}
# "Shop by step" buttons  ->  keywords searched in Category + Product_Name
STEP_KEYWORDS = {
    "cleanse":   ["cleanser", "cleansing", "face wash", "wash", "micellar", "foam", "makeup remover", "cleansing oil", "cleansing balm"],
    "treat":     ["serum", "treatment", "essence", "ampoule", "toner", "booster", "retinol", "spot", "exfoliant", "peel", "acid", "eye"],
    "moisturize": ["moisturizer", "moisturiser", "cream", "lotion", "gel", "emulsion", "hydrating", "balm"],
    "protect":   ["sunscreen", "sun screen", "spf", "sun protection", "uv", "sunblock", "sun"],
    "mask":      ["mask", "masque", "sheet", "sleeping", "overnight", "pack"],
}
STOP = set("a an the for with and or to of my i need want something product products skin face help me is are in on".split())
QUERY_STOP = STOP | {"good", "best", "please", "looking"}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def clean(v):
    if pd.isna(v): return ""
    s = str(v).strip()
    return "" if s.lower() in {"unknown", "unknown brand", "unknown category", "unknown ingredients",
                               "unknown frequency", "nan", "none"} else s

def normalize_text(s):
    s = str(s or "").lower()
    s = re.sub(r"[^a-z0-9% ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def stem(w):
    """Very light stemmer (plural -> singular) so 'pimples' matches 'pimple'."""
    if len(w) > 4 and w.endswith("ies"): return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w.endswith("us"): return w[:-1]
    return w

def tokenize(s):
    return [stem(t) for t in normalize_text(s).split()]

def lookup(table, term):
    for key in (term, term + "s", term[:-1] if term.endswith("s") else None):
        if key and key in table: return table[key]
    return []

def row_text(row):
    # Field weighting: name / concern / category are repeated so they count more.
    name, concern, cat = clean(row.get("Product_Name")), clean(row.get("Concern")), clean(row.get("Category"))
    return " ".join([name, name, concern, concern, cat, cat, clean(row.get("Skin_Type")),
                     clean(row.get("Brand")), clean(row.get("Ingredients")), clean(row.get("Product_Size"))])

# ----------------------------------------------------------------------------
# Data loading (only products that have a usable picture are kept)
# ----------------------------------------------------------------------------
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        from datasets import load_dataset
        df = load_dataset(DATASET_ID, split="train").to_pandas()
        os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    if MAX_ROWS > 0: df = df.head(MAX_ROWS).copy()
    for col in ["Product_Name", "Skin_Type", "Concern", "Category", "product_url", "product_pic", "Brand",
                "Usage_Frequency", "Product_Size", "Gender_Target", "Packaging_Type", "Ingredients", "Country_of_Origin"]:
        if col not in df: df[col] = ""
        df[col] = df[col].map(clean)
    for col in ["Price_USD", "Rating", "Number_of_Reviews"]:
        if col not in df: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "Cruelty_Free" not in df: df["Cruelty_Free"] = False
    df["Cruelty_Free"] = df["Cruelty_Free"].fillna(False).astype(bool)
    df = df[df["Product_Name"].str.len() > 1].copy()

    # ---- keep ONLY products with a picture URL ----
    pics = df["product_pic"].str.strip()
    ok = pics.str.match(r"^https?://[^\s<>\"']+$", case=False, na=False) & \
         ~pics.str.lower().str.contains(r"placeholder|no[-_ ]?image|default[-_ ]?image", regex=True, na=False)
    df = df[ok].copy()
    df["_q"] = df["product_url"].str.startswith("http", na=False).astype(int)
    df = df.sort_values("_q", ascending=False).drop_duplicates(subset=["Product_Name", "Brand"], keep="first").drop(columns=["_q"])
    df = df.reset_index(drop=True)
    df["id"] = df.index.astype(str)
    df["search_text"] = df.apply(row_text, axis=1)
    return df

df = load_data()
N = len(df)
# Pre-computed normalised columns (fast vectorised filtering)
norm_skin = df["Skin_Type"].map(normalize_text)
norm_concern = df["Concern"].map(normalize_text)
norm_cat = df["Category"].map(normalize_text)
norm_name = df["Product_Name"].map(normalize_text)
norm_text = df["search_text"].map(normalize_text)

documents = [tokenize(t) for t in df["search_text"]]
bm25 = BM25Okapi(documents)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
tfidf_matrix = vectorizer.fit_transform([" ".join(d) for d in documents])
# quality prior in [0,1] from rating + number of reviews
_max_rev = max(1.0, float(df["Number_of_Reviews"].max()))
quality = ((df["Rating"].clip(0, 5) / 5) * 0.7 +
           (np.log1p(df["Number_of_Reviews"]) / math.log1p(_max_rev)) * 0.3).to_numpy()

feedback_memory = defaultdict(Counter)
negative_feedback = defaultdict(set)
positive_feedback = defaultdict(set)
feedback_lock = threading.Lock()

# ----------------------------------------------------------------------------
# STEP 1 + 2 + 3 : identify terms, expand terms, expansion methods
# ----------------------------------------------------------------------------
def identify_terms(query):
    """Step 1: query terms = meaningful words + known multi-word phrases."""
    q = normalize_text(query)
    words = [w for w in q.split() if w not in QUERY_STOP]
    phrases = []
    for table in (SYNONYMS, RELATED, CONCEPTS, CONCERN_ALIASES):
        for key in table:
            if " " in key and f" {key} " in f" {q} ": phrases.append(key)
    return list(dict.fromkeys(words + phrases))

def _uniq(items, exclude=()):
    out, seen = [], set(exclude)
    for t in items:
        t = normalize_text(t)
        if t and t not in seen: seen.add(t); out.append(t)
    return out

def expand_query(query, feedback_terms=None, auto_terms=None):
    """Steps 2-3: returns a dict with the terms produced by each expansion method."""
    original = identify_terms(query)
    syn = _uniq([s for t in original for s in lookup(SYNONYMS, t)], original)
    rel = _uniq([r for t in original + syn for r in lookup(RELATED, t)], original + syn)
    con = _uniq([c for t in original + syn + rel for c in lookup(CONCEPTS, t)], original + syn + rel)
    known = original + syn + rel + con
    fb = _uniq(feedback_terms or [], known)
    auto = _uniq(auto_terms or [], known + fb)
    return {"original": original, "synonyms": syn, "related": rel, "concepts": con,
            "feedback": fb, "automatic": auto}

EXPANSION_WEIGHTS = {"original": 3.0, "synonyms": 1.5, "related": 1.0, "concepts": 0.7, "feedback": 1.0, "automatic": 0.7}

def weighted_tokens(expansion):
    """Original terms count more than expansion terms (token -> weight)."""
    w = {}
    for method, weight in EXPANSION_WEIGHTS.items():
        for term in expansion.get(method, []):
            for tok in tokenize(term):
                if tok in QUERY_STOP: continue
                w[tok] = max(w.get(tok, 0), weight)
    return w

def flat_terms(expansion):
    return [t for k in EXPANSION_WEIGHTS for t in expansion.get(k, [])]

# ----------------------------------------------------------------------------
# Scoring: BM25 / TF-IDF over the whole collection
# ----------------------------------------------------------------------------
def bm25_scores(wtok):
    s = np.zeros(N)
    for tok, weight in wtok.items():
        s += weight * np.asarray(bm25.get_scores([tok]), dtype=float)
    s = np.clip(s, 0, None)
    return s / s.max() if s.max() > 0 else s

def tfidf_scores(wtok):
    text = " ".join(" ".join([t] * max(1, round(w * 2))) for t, w in wtok.items())
    if not text.strip(): return np.zeros(N)
    s = cosine_similarity(vectorizer.transform([text]), tfidf_matrix).ravel()
    return s / s.max() if s.max() > 0 else s

def useful_term(tok):
    """Skip stop-words, numbers/sizes (50ml) and very common words (water...)."""
    return (tok not in STOP and len(tok) > 3 and not any(ch.isdigit() for ch in tok)
            and bm25.idf.get(tok, 0.0) > 1.0)

def automatic_expansion(first_pass, exclude, k=5, n_terms=5):
    """Automatic expansion = pseudo-relevance feedback: take the top-k documents
    of a first retrieval and add the terms they share (weighted by IDF)."""
    top = [i for i in np.argsort(-first_pass)[:k] if first_pass[i] > 0]
    docs_with, weight = Counter(), Counter()
    for i in top:
        for tok in set(documents[i]):
            if tok in exclude or not useful_term(tok): continue
            docs_with[tok] += 1
            weight[tok] += bm25.idf.get(tok, 0.0)
    return [t for t, _ in weight.most_common() if docs_with[t] >= 2][:n_terms]

# ----------------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------------
def _contains_any(series, words):
    pat = "|".join(re.escape(w) for w in words)
    return series.str.contains(pat, regex=True, na=False)

def step_mask(step):
    kws = STEP_KEYWORDS[step]
    cat_hit = _contains_any(norm_cat, kws)
    # fall back to the product name only when the category column doesn't say anything
    name_hit = _contains_any(norm_name, kws) & (norm_cat == "")
    return (cat_hit | name_hit).to_numpy()

def filter_mask(skin_type="", concern="", category=""):
    mask = np.ones(N, dtype=bool)
    if skin_type:
        q = normalize_text(skin_type)
        mask &= (norm_skin.str.contains(q, regex=False) | norm_skin.str.contains(r"\ball\b", regex=True)).to_numpy()
    if concern:
        q = normalize_text(concern)
        words = CONCERN_ALIASES.get(q, [q])
        mask &= (_contains_any(norm_concern, words) | _contains_any(norm_text, words)).to_numpy()
    if category and category.lower() != "all":
        q = normalize_text(category)
        if q in STEP_KEYWORDS: mask &= step_mask(q)
        else: mask &= (norm_cat == q).to_numpy()
    return mask

# ----------------------------------------------------------------------------
# Serialisation
# ----------------------------------------------------------------------------
FIELDS = ["id", "Product_Name", "Skin_Type", "Concern", "Category", "product_url", "product_pic", "Brand",
          "Usage_Frequency", "Product_Size", "Gender_Target", "Packaging_Type", "Ingredients", "Country_of_Origin"]

def serialize(i, score=None):
    r = df.iloc[int(i)]
    row = {f: str(r[f]) for f in FIELDS}
    row["Price_USD"] = float(r["Price_USD"] or 0)
    row["Rating"] = float(r["Rating"] or 0)
    row["Number_of_Reviews"] = int(r["Number_of_Reviews"] or 0)
    row["Cruelty_Free"] = bool(r["Cruelty_Free"])
    if score is not None: row["score"] = float(score)
    return row

# ----------------------------------------------------------------------------
# STEP 4 : search with re-ranking
# ----------------------------------------------------------------------------
def search_products(query, method="bm25", skin_type="", concern="", category="", limit=60):
    qkey = normalize_text(query)
    with feedback_lock:
        learned = [w for w, _ in feedback_memory[qkey].most_common(8)]
        disliked = set(negative_feedback[qkey])
    mask = filter_mask(skin_type, concern, category)

    # first pass (original + knowledge-based expansion + feedback) ...
    exp = expand_query(query, feedback_terms=learned)
    wtok = weighted_tokens(exp)
    first = bm25_scores(wtok) if method == "bm25" else tfidf_scores(wtok)
    # ... automatic expansion (pseudo-relevance feedback) and second pass
    auto = automatic_expansion(first * mask, exclude=set(wtok))
    exp = expand_query(query, feedback_terms=learned, auto_terms=auto)
    wtok = weighted_tokens(exp)
    bm, tf = bm25_scores(wtok), tfidf_scores(wtok)

    # re-ranking: primary model + other model + quality prior
    primary, secondary = (bm, tf) if method == "bm25" else (tf, bm)
    final = 0.65 * primary + 0.25 * secondary + 0.10 * quality * (primary > 0)
    for pid in disliked: final[int(pid)] *= 0.05
    final = np.where(mask & (primary + secondary > 0), final, 0)
    order = np.argsort(-final)[:limit]
    results = [serialize(i, min(1.0, final[i])) for i in order if final[i] > 0]
    return results, exp

# ----------------------------------------------------------------------------
# Content-based recommendation
# ----------------------------------------------------------------------------
def recommend(skin_type="", concern="", query="", category="", limit=8):
    parts = []  # (weight, score-array)
    profile = {}
    if skin_type:
        for t in tokenize(skin_type): profile[t] = max(profile.get(t, 0), 2.0)
    wanted = " ".join(x for x in [concern, query] if x)
    if wanted:
        for t, w in weighted_tokens(expand_query(wanted)).items(): profile[t] = max(profile.get(t, 0), w)
    if concern:
        for t in tokenize(concern): profile[t] = max(profile.get(t, 0), 3.0)
    cos = tfidf_scores(profile) if profile else np.zeros(N)
    parts.append((0.40, cos))

    skin_match = np.ones(N)
    if skin_type:
        q = normalize_text(skin_type)
        exact = norm_skin.str.contains(q, regex=False).to_numpy()
        generic = norm_skin.str.contains(r"\ball\b", regex=True).to_numpy()
        skin_match = np.where(exact, 1.0, np.where(generic, 0.6, 0.0))
        parts.append((0.25, skin_match))
    if concern:
        words = CONCERN_ALIASES.get(normalize_text(concern), [normalize_text(concern)])
        in_field = _contains_any(norm_concern, words).to_numpy()
        in_text = _contains_any(norm_text, words).to_numpy()
        parts.append((0.35, np.where(in_field, 1.0, np.where(in_text, 0.5, 0.0))))
    total_w = sum(w for w, _ in parts)
    score = sum(w * s for w, s in parts) / total_w
    score = score * (0.9 + 0.1 * quality)          # quality prior
    if skin_type: score = score * np.where(skin_match > 0, 1.0, 0.4)   # wrong skin type is demoted
    if category: score = np.where(filter_mask(category=category), score, 0)
    if not profile and not skin_type and not concern: score = quality.copy()
    order = np.argsort(-score)[:limit]
    return [serialize(i, min(1.0, score[i])) for i in order if score[i] > 0]

# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
def _int(v, default): 
    try: return int(v)
    except (TypeError, ValueError): return default

@app.route("/")
def home():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "dataset": DATASET_ID, "products": N})

@app.get("/api/stats")
def stats():
    return jsonify({"products": N, "categories": int(df["Category"].nunique()),
                    "skin_types": int(df["Skin_Type"].nunique()), "products_with_images": N})

@app.get("/api/meta")
def meta():
    cats = [c for c, _ in Counter(df["Category"]).most_common() if c]
    concerns = Counter(x.strip() for v in df["Concern"] for x in v.split(",") if x.strip())
    return jsonify({"categories": cats, "concerns": [c for c, _ in concerns.most_common(14)]})

@app.get("/api/products")
def products():
    """Browse (no query): filtered by step/category/concern/skin type, best rated first."""
    a = request.args
    limit = min(500, max(1, _int(a.get("limit"), 120)))
    offset = max(0, _int(a.get("offset"), 0))
    mask = filter_mask(a.get("skin_type", ""), a.get("concern", ""), a.get("category", ""))
    idx = np.where(mask)[0]
    idx = idx[np.argsort(-quality[idx], kind="stable")]
    return jsonify({"products": [serialize(i) for i in idx[offset:offset + limit]], "total": int(len(idx))})

@app.post("/api/search")
def search():
    data = request.get_json(force=True, silent=True) or {}
    query = clean(data.get("query", ""))
    if not query: return jsonify({"results": [], "total": 0, "expanded_terms": []})
    method = str(data.get("method", "bm25")).lower()
    if method not in {"bm25", "tfidf"}: method = "bm25"
    results, exp = search_products(query, method, data.get("skin_type", ""), data.get("concern", ""),
                                   data.get("category", ""), min(150, max(1, _int(data.get("limit"), 60))))
    return jsonify({"results": results, "total": len(results), "method": method,
                    "expanded_terms": flat_terms(exp), "expansion": exp})

@app.post("/api/recommend")
def recommendations():
    data = request.get_json(force=True, silent=True) or {}
    recs = recommend(data.get("skin_type", ""), data.get("concern", ""), data.get("query", ""),
                     data.get("category", ""), min(60, max(1, _int(data.get("limit"), 8))))
    return jsonify({"recommendations": recs})

@app.post("/api/feedback")
def feedback():
    """Feedback-based expansion (Rocchio-style): terms of 'relevant' products are
    added to this query next time; 'irrelevant' products are pushed down."""
    data = request.get_json(force=True, silent=True) or {}
    query = normalize_text(data.get("query", ""))
    ids = [str(x) for x in data.get("product_ids", [])]
    kind = data.get("feedback", "")
    if not query or kind not in {"relevant", "irrelevant"}:
        return jsonify({"error": "Invalid feedback"}), 400
    valid = [p for p in ids if p.isdigit() and int(p) < N]
    if kind == "relevant":
        with feedback_lock:
            for pid in valid:
                positive_feedback[query].add(pid)
                negative_feedback[query].discard(pid)
                for tok in set(documents[int(pid)]):
                    if not useful_term(tok): continue
                    feedback_memory[query][tok] += bm25.idf.get(tok, 0.0)
            extra = [w for w, _ in feedback_memory[query].most_common(6)]
        return jsonify({"message": "Got it — results will now lean towards products like these ♡",
                        "expanded_query": " ".join([query] + extra)})
    with feedback_lock:
        for pid in valid:
            negative_feedback[query].add(pid)
            positive_feedback[query].discard(pid)
    return jsonify({"message": "Thanks — those products will be ranked lower for this search."})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)