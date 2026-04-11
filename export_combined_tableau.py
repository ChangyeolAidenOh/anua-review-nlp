import os
import pandas as pd

# Combines main LDA and TF-IDF LDA results into a single Tableau-ready CSV.
# Use the 'pipeline' column to compare both approaches side by side.

if __name__ == "__main__":

    # Load main LDA results
    main_path = "anua_tableau_master.csv"
    tfidf_path = "anua_tableau_tfidf.csv"

    dfs = []

    if os.path.exists(main_path):
        df_main = pd.read_csv(main_path)
        df_main["pipeline"] = "main_lda"
        dfs.append(df_main)
        print(f"Loaded main LDA: {len(df_main)} rows")
    else:
        print(f"Warning: {main_path} not found — run export_tableau.py first")

    if os.path.exists(tfidf_path):
        df_tfidf = pd.read_csv(tfidf_path)
        df_tfidf["pipeline"] = "tfidf_lda"
        dfs.append(df_tfidf)
        print(f"Loaded TF-IDF LDA: {len(df_tfidf)} rows")
    else:
        print(f"Warning: {tfidf_path} not found — run export_tableau_tfidf.py first")

    if not dfs:
        raise FileNotFoundError("No CSV files found. Run export scripts first.")

    df_combined = pd.concat(dfs, ignore_index=True)

    output_path = "anua_tableau_combined.csv"
    df_combined.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(df_combined)} rows)")
    print(f"Pipelines: {df_combined['pipeline'].value_counts().to_dict()}")
    print(df_combined.head(5).to_string(index=False))