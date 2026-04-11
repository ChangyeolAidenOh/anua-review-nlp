import os
import pandas as pd

# Load TF-IDF filtered LDA topic_summary from both products
# and expand to row-per-keyword format for Tableau exploration.

if __name__ == "__main__":
    products = [
        ("cleansing_oil", "outputs_tfidf_cleansing_oil/topic_summary.csv"),
        ("toner",         "outputs_tfidf_toner/topic_summary.csv"),
    ]

    dfs = []
    for product_name, path in products:
        if not os.path.exists(path):
            print(f"Warning: {path} not found — skipping")
            continue
        df = pd.read_csv(path)
        df["product"] = product_name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No topic_summary.csv found. Run LDA_tfidf_pipeline.py first.")

    df_summary = pd.concat(dfs, ignore_index=True)

    # Expand keywords to one row per keyword
    rows = []
    for _, record in df_summary.iterrows():
        if pd.isnull(record.get("keywords")):
            continue

        topics = str(record["keywords"]).split(" | ")
        for topic_id, topic_str in enumerate(topics, start=1):
            keywords = [kw.strip() for kw in topic_str.split(",") if kw.strip()]
            for rank, keyword in enumerate(keywords, start=1):
                rows.append({
                    "product":            record["product"],
                    "preprocessing_mode": record["mode"],
                    "star_rating":        record["rating"],
                    "optimal_k":          record["optimal_k"],
                    "coherence_score":    record["coherence"],
                    "num_reviews":        record["num_reviews"],
                    "topic_id":           topic_id,
                    "keyword":            keyword,
                    "keyword_rank":       rank,
                    "pipeline":           "tfidf_lda",  # distinguishes from main LDA
                })

    df_tableau = pd.DataFrame(rows)

    output_path = "anua_tableau_tfidf.csv"
    df_tableau.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(df_tableau)} rows)")
    print(df_tableau.head(10).to_string(index=False))