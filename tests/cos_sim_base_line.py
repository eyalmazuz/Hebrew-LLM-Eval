from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

# Load your dataset
df = pd.read_csv("./tests/output/labeled_data_with_scores.csv")

# Example columns
sum_col = "Sentence in Summary"
art_col = "Best Match Sentences From Article"
# sum_col = "Summary"
# art_col = "Article"

# Handle NaNs
df[sum_col] = df[sum_col].fillna("")
df[art_col] = df[art_col].fillna("")

# Create TF-IDF vectorizer (English example; set analyzer='word' or 'char')
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5))

# Fit on all text (optional: you can fit only on articles)
vectorizer.fit(df[art_col].tolist() + df[sum_col].tolist())

# Transform texts into vectors
article_vecs = vectorizer.transform(df[art_col])
summary_vecs = vectorizer.transform(df[sum_col])

# Compute cosine similarity row by row
similarities = []
for i in range(len(df)):
    sim = cosine_similarity(article_vecs[i], summary_vecs[i])[0][0]
    similarities.append(sim)

df["TFIDF_COS_SIM"] = similarities
agg_cols = ["score", "TFIDF_COS_SIM"]
# If you want, you can group by Article to get mean similarity per article
df_article = df.groupby("Article", as_index=False)[agg_cols].mean()

# 7) Correlations (coefficients + p-values)
cols = ["score", "TFIDF_COS_SIM"]

def corr_table_with_pvals(data, cols, method="pearson"):
    out = pd.DataFrame(index=cols, columns=cols, dtype=object)
    for i in cols:
        for j in cols:
            if method == "pearson":
                r, p = pearsonr(data[i], data[j])
            elif method == "spearman":
                r, p = spearmanr(data[i], data[j])
            elif method == "kendall":
                r, p = kendalltau(data[i], data[j])
            out.loc[i, j] = f"{r:.3f} (p={p:.3g})"
    return out

print("Pearson:\n", corr_table_with_pvals(df_article, cols, "pearson"), "\n")
print("Spearman:\n", corr_table_with_pvals(df_article, cols, "spearman"), "\n")
print("Kendall:\n", corr_table_with_pvals(df_article, cols, "kendall"), "\n")

# 8) Save for later
# df.to_csv("./tests/output/dataset_with_emb_sims_sent.csv", index=False)
df_article.to_csv("./tests/output/dataset_with_cos_sims_article.csv", index=False)
