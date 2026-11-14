# ✿ Blushed : Skincare Search & Recommendation System
A small search engine and recommender for cosmetic products

Dataset: [`Yaminii/skincare1`](https://huggingface.co/datasets/Yaminii/skincare1) (only products that have a picture are shown).

## Features
- **Search** with query expansion and re-ranking (BM25 or TF-IDF, selectable in the page).
- **Shop by step** (Cleanse, Treat, Moisturize, Protect, Mask) and category / concern filters.
- **Content-based recommendations** from the user's skin type and concern ("Your glow match").
- **Relevance feedback**: "More like these" / "Not relevant" buttons improve the next search.

## How the query-expansion pipeline works
| Step | Function (`app.py`) | What it does |
|---|---|---|
| 1. Identify query terms | `identify_terms()` | Removes stop-words, detects phrases such as "dark spots" |
| 2. Expand terms | `expand_query()` | Builds the expanded query, one list per method |
| 3a. Synonym expansion | `SYNONYMS` | pimples → acne, breakouts, blemishes |
| 3b. Related-term expansion | `RELATED` | acne → salicylic acid, niacinamide |
| 3c. Concept expansion | `CONCEPTS` | hydration → moisture retention, skin barrier support |
| 3d. Feedback-based expansion | `/api/feedback` | Terms of products marked relevant are added (Rocchio-style) |
| 3e. Automatic expansion | `automatic_expansion()` | Pseudo-relevance feedback: shared terms of the top-5 first-pass results |
| 4. Re-ranking | `search_products()` | First pass with BM25 or TF-IDF, then hybrid score: 65% chosen model + 25% other model + 10% rating/reviews, minus penalty for "not relevant" products |

Original query terms are weighted 3×, synonyms 1.5×, related terms 1×, concepts and automatic terms 0.7×.

**Recommendation score** = 40% TF-IDF similarity to the user profile + 25% skin-type match + 35% concern match, multiplied by a small rating/reviews quality factor.

## Run locally
```bash
pip install -r requirements.txt
pip install datasets          # only needed once, to download the dataset
python app.py
```
Open http://127.0.0.1:5000. The first run downloads the dataset into `data/skincare1.csv`.

Project structure:
```
app.py
requirements.txt
templates/index.html
data/skincare1.csv      (created on first run)
```



## API
| Route | Description |
|---|---|
| `GET /api/products` | Browse with `category`, `concern`, `skin_type`, `limit`, `offset` |
| `POST /api/search` | `{query, method: "bm25"\|"tfidf", category, concern}` |
| `POST /api/recommend` | `{skin_type, concern, query}` |
| `POST /api/feedback` | `{query, product_ids, feedback: "relevant"\|"irrelevant"}` |
| `GET /api/meta` | Categories and concerns for the filter chips |
