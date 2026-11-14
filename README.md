# blushed — Skincare Search & Recommendation System

This project connects the supplied `blushed` HTML interface to the Hugging Face
`Yaminii/skincare1` dataset and implements the Information Retrieval Exercise 2.

## Implemented
- Dataset loading from Hugging Face (`train` split), CSV caching, cleaning and deduplication.
- Product images displayed directly from the dataset's `product_pic` URLs.
- Browser-side image loading checks: products with broken, blocked, or missing images are excluded from the visible shop, search results, and recommendations.
- The large product-image arches below the hero buttons are removed to keep the landing page clean.
- Query term identification through text normalization/tokenization.
- Query expansion: synonym, related-term, concept, corpus-based automatic expansion.
- Feedback-based expansion from products marked relevant.
- BM25 and TF-IDF search ranking (switchable in the interface).
- Content-based recommendations using TF-IDF cosine similarity and skin-profile matching.
- Flask JSON endpoints and the original pastel `blushed` interface.

## Run on Windows
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Start the application:

   ```powershell
   python app.py
   ```

6. Open http://127.0.0.1:5000

The first launch downloads the Hugging Face dataset and caches it as
`data/skincare1.csv`. Internet access is needed only for this initial download
and for loading externally hosted product pictures.

## API
- `GET /api/health`
- `GET /api/stats`
- `GET /api/products?limit=120`
- `POST /api/search` — `{ "query": "pimples for oily skin", "method": "bm25" }`
- `POST /api/recommend` — `{ "skin_type": "Oily", "concern": "Acne" }`
- `POST /api/feedback` — relevant/irrelevant feedback

## Notes and limitations
- The dataset is third-party data. Some values are explicitly `Unknown`, and
  external image URLs may be broken or blocked by their hosts; the UI falls back
  gracefully when an image fails.
- The dataset includes skincare products, not a verified medical treatment
  database. Recommendations are information-retrieval matches, not medical advice.
- Feedback-based expansion is an in-memory demonstration and resets when the
  server restarts. It is intended to demonstrate relevance feedback for the exercise.
- The shopping bag is a frontend demo; checkout/payment is not implemented.
- Scores from BM25 and TF-IDF are method-specific relevance scores, not calibrated
  probabilities. Do not compare their raw values as if they shared one scale.

## Project structure
```
blushed_skincare_project/
  app.py
  requirements.txt
  templates/
    index.html
  README.md
```


### Image-only product catalog

The app automatically excludes dataset rows whose `product_pic` field is empty, `Unknown`, or not an HTTP(S) image URL. Search results, category/skin filters, recommendations, and product counts therefore use only products with an image URL. This checks that an image URL is present; it cannot guarantee that the remote host will always serve the image.

## Image display behavior
The backend removes records with empty or malformed `product_pic` values. The frontend then tests whether each remote image can actually load before rendering that product. This applies to the shop, search results, and recommendations. Because images are hosted by third parties, availability can still change after the check.
