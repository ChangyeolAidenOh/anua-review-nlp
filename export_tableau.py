import os
import pandas as pd

# Load topic_summary.csv from both products and merge into
# a single row-per-keyword CSV optimized for Tableau exploration.

if __name__ == "__main__":

    # Load both product summaries
    dfs = []
    for product_name, path in [
        ("cleansing_oil", "outputs_cleansing_oil/topic_summary.csv"),
        ("toner",         "outputs_toner/topic_summary.csv"),
    ]:
        if not os.path.exists(path):
            print(f"Warning: {path} not found — skipping")
            continue
        df = pd.read_csv(path)
        df["product"] = product_name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No topic_summary.csv found. Run LDA_pipeline.py first.")

    df_summary = pd.concat(dfs, ignore_index=True)

    # Expand keywords to one row per keyword
    # Original: "acne, pore | oil, gentle"  (topics by |, keywords by ,)
    # Expanded: one row per keyword with topic_id and keyword_rank
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
                    "keyword_rank":       rank,  # 1 = most representative in topic
                })

    df_tableau = pd.DataFrame(rows)

    output_path = "anua_tableau_master.csv"
    df_tableau.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(df_tableau)} rows)")
    print(df_tableau.head(10).to_string(index=False))
