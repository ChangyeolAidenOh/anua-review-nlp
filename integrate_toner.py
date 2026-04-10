import pandas as pd
import os

# 별점별 파일명 (예시: "Review-B08CMS8P67-one_star.xlsx" 등)
file_mapping = {
    1: "Review-B08CMS8P67-one_star.xlsx",
    2: "Review-B08CMS8P67-two_star.xlsx",
    3: "Review-B08CMS8P67-three_star.xlsx",
    4: "Review-B08CMS8P67-four_star.xlsx",
    5: "Review-B08CMS8P67-five_star.xlsx",
}

all_reviews = []

for rating, filename in file_mapping.items():
    df = pd.read_excel(filename)
    df["rating"] = rating  # 별점 추가
    all_reviews.append(df)

# 통합 저장
combined_df = pd.concat(all_reviews, ignore_index=True)
combined_df.to_csv("anua_toner_reviews.csv", index=False)
