import pandas as pd

# 파일 경로와 별점 매핑
file_rating_pairs = [
    ("Review-B0BN2PX8V3-one_star.xlsx", 1),
    ("Review-B0BN2PX8V3-two_star.xlsx", 2),
    ("Review-B0BN2PX8V3-three_star.xlsx", 3),
    ("Review-B0BN2PX8V3-four_star.xlsx", 4),
    ("Review-B0BN2PX8V3-five_star.xlsx", 5),
]

# 통합 리스트
dfs = []

for file, rating in file_rating_pairs:
    df = pd.read_excel(file)
    df["rating"] = rating
    dfs.append(df)

# 하나로 합치기
full_df = pd.concat(dfs, ignore_index=True)

# 저장
full_df.to_csv("anua_cleansing_oil_reviews.csv", index=False)
