import os
import pandas as pd
import ast

# Loads Laplace pipeline summary CSVs from both versions (v1, v2)
# and expands segment data into row-per-segment format for Tableau.

if __name__ == "__main__":

    sources = [
        ("cleansing_oil", "v1", "outputs_laplace_v1_cleansing_oil/laplace_summary.csv"),
        ("toner",         "v1", "outputs_laplace_v1_toner/laplace_summary.csv"),
        ("cleansing_oil", "v2", "outputs_laplace_cleansing_oil/laplace_summary.csv"),
        ("toner",         "v2", "outputs_laplace_toner/laplace_summary.csv"),
    ]

    rows = []

    for product_name, version, path in sources:
        if not os.path.exists(path):
            print(f"Warning: {path} not found — skipping")
            continue

        df = pd.read_csv(path)
        print(f"Loaded {path}: {len(df)} rows")

        for _, record in df.iterrows():
            try:
                # Parse segments string: [(start, end, label), ...]
                segments = ast.literal_eval(record["segments"])
                for seg_id, (start, end, label) in enumerate(segments, start=1):
                    rows.append({
                        "product":         record["product"],
                        "star_rating":     record["rating"],
                        "n_reviews":       record["n_reviews"],
                        "version":         version,
                        "segment_id":      seg_id,
                        "segment_start":   round(start, 2),
                        "segment_end":     round(end, 2),
                        "segment_length":  round(end - start, 2),
                        "segment_label":   label,  # UP / DOWN / FLAT
                        "topic_mode":      record.get("topic_mode", ""),
                        "sentiment_backend": record.get("sentiment_backend", ""),
                    })
            except Exception as e:
                print(f"  Warning: could not parse segments for {product_name} rating {record.get('rating')} — {e}")

    if not rows:
        raise ValueError("No segment data found. Run laplace pipeline scripts first.")

    df_tableau = pd.DataFrame(rows)

    output_path = "anua_tableau_laplace.csv"
    df_tableau.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(df_tableau)} rows)")
    print(f"Label distribution:\n{df_tableau['segment_label'].value_counts().to_dict()}")
    print(df_tableau.head(10).to_string(index=False))