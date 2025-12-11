"""
Overall ranking across models

→ Friedman test
→ Nemenyi post-hoc test
"""
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
from autorank import autorank, plot_stats, create_report
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# # ---- Load individual CSVs ----
# df_base = pd.read_csv("./tests/output/dataset_with_baselines.csv")
# df_llm = pd.read_csv("./tests/output/labeled_data_with_LLM_scores.csv")
# df_external = pd.read_csv("./tests/output/external_model_baseline.csv")
# df_ours = pd.read_csv("./scripts/output/test_topic2_merged.csv")

# # print("External CSV columns:", df_base.columns.tolist())


# # ---- Standardize column names ----
# df_base = df_base.rename(columns={
#     "EMB_SIM": "EMB",
#     "TFIDF_COS_SIM": "TFIDF",
#     "ROUGE-1": "ROUGE1",
#     "ROUGE-2": "ROUGE2",
#     "ROUGE-L": "ROUGEL",
#     "BLEU": "BLEU",
# })

# df_llm = df_llm.rename(columns={"LLM_score": "LLM"})
# df_external = df_external.rename(columns={"stance_shift": "External"})
# df_ours = df_ours.rename(columns={"emd_score": "Ours"})

# # print("External CSV columns:", df_base.columns.tolist())


# # ---- KEEP ONLY NEEDED COLUMNS ----
# df_base = df_base[["Article", "Summary", "EMB", "TFIDF", "ROUGE1", "ROUGE2", "ROUGEL", "BLEU"]]
# df_llm = df_llm[["Article", "Summary", "LLM"]]
# df_external = df_external[["Article", "Summary", "External"]]
# df_ours = df_ours[["Article", "Summary", "Ours"]]

# # ---- Merge ----
# df = df_base.merge(df_llm, on=["Article", "Summary"], how="inner")
# df = df.merge(df_external, on=["Article", "Summary"], how="inner")
# df = df.merge(df_ours, on=["Article", "Summary"], how="inner")

# print("Merged shape:", df.shape)
# df.to_csv("./tests/test_data/article_level_merged_scores.csv", index=False)
# print("Columns:", df.columns.tolist())

# --------------------------------------------------------------------------------------------------------------------------------------------
# Friedman + Nemenyi test 
# --------------------------------------------------------------------------------------------------------------------------------------------
# # ---- Load merged data ----
# df = pd.read_csv("./tests/test_data/article_level_merged_scores.csv")
# # print("df:\n", df.head())

# # ---- Select model columns ----
# model_cols = ["Ours", "BLEU", "ROUGE1", "ROUGE2", "ROUGEL", "EMB", "TFIDF", "LLM", "External"]

# # Drop NA rows
# df = df.dropna(subset=model_cols)

# # ---- Compute ranks (higher = better) ----
# ranks = df[model_cols].rank(axis=1, ascending=False)

# # ---- Friedman test ----
# stat, p = stats.friedmanchisquare(*[ranks[col] for col in model_cols])
# print("\nFriedman statistic:", stat)
# print("Friedman p-value:", p)

# # ---- Nemenyi post-hoc (scikit-posthocs) ----
# nemenyi = sp.posthoc_nemenyi_friedman(ranks)
# print("\nNemenyi post-hoc matrix:")
# print(nemenyi)

# # ---- autorank: performs ranking analysis & CD diagram ----
# results = autorank(ranks, alpha=0.05, verbose=False)

# print("\n=== Autorank results ===")
# print(results)

# # CD diagram
# plot_stats(results)
# plt.savefig("./tests/output/cd_diagram.png", bbox_inches='tight')


# # Text report
# report = create_report(results)
# print("\n=== Autorank report ===")
# print(report)

# --------------------------------------------------------------------------------------------------------------------------------------------
# Wilcoxon signed rank test + Holm correction 
# --------------------------------------------------------------------------------------------------------------------------------------------
# ---- Load merged data ----
df = pd.read_csv("./tests/test_data/article_level_merged_scores.csv")
# print("df:\n", df.head())
model_cols = ["Ours", "BLEU", "ROUGE1", "ROUGE2", "ROUGEL", "EMB", "TFIDF", "LLM", "External"]
df = df.dropna(subset=model_cols)

# Prepare pairwise comparisons
pairs = [(i, j) for i in range(len(model_cols)) for j in range(i+1, len(model_cols))]
pvals = []

for i, j in pairs:
    x = df[model_cols[i]]
    y = df[model_cols[j]]

    # Handle zero-difference case
    if (x != y).any():
        stat, p = wilcoxon(x, y)
    else:
        p = 1.0  # No difference
    pvals.append(p)

# Holm correction
reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='holm')

# Show results
for k, (i, j) in enumerate(pairs):
    print(f"{model_cols[i]} vs {model_cols[j]}: raw p={pvals[k]:.3e}, Holm-corrected p={pvals_corrected[k]:.3e}, reject={reject[k]}")

# --- Count wins per model ---
wins = {model: 0 for model in model_cols}

for k, (i, j) in enumerate(pairs):
    if reject[k]:  # significant difference
        # higher median wins
        median_i = df[model_cols[i]].median()
        median_j = df[model_cols[j]].median()
        if median_i > median_j:
            wins[model_cols[i]] += 1
        else:
            wins[model_cols[j]] += 1

# --- Rank models by wins ---
ranking = sorted(wins.items(), key=lambda x: x[1], reverse=True)
print("\nModel ranking based on pairwise Wilcoxon + Holm correction:")
for rank, (model, score) in enumerate(ranking, 1):
    print(f"{rank}. {model} (wins: {score})")