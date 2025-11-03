# pip install -U sentence-transformers scikit-learn pandas numpy scipy
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr, spearmanr, kendalltau
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
import tiktoken


_ = load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

df = pd.read_csv("./tests/output/labeled_data_with_scores.csv")
# agg_cols = ["score"]
# article_df = df.groupby("Article", as_index=False)[agg_cols].mean()
# art_col = "Article"
# model_name = "text-embedding-3-small"
# MAX_TOKENS = 8192

# # Load the correct tokenizer for this model
# enc = tiktoken.encoding_for_model(model_name)

# # Count tokens for each article
# article_df["article_token_count"] = article_df[art_col].apply(lambda x: len(enc.encode(x)))

# # Find long articles
# too_long = article_df[article_df["article_token_count"] > MAX_TOKENS]

# print(f"Total articles: {len(article_df)}")
# print(f"Articles exceeding {MAX_TOKENS} tokens: {len(too_long)}")
# print()

# if not too_long.empty:
#     print("Examples of long articles:")
#     display_cols = [art_col, "article_token_count"]
#     print(too_long[display_cols].head(10).to_string(index=False))

# model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# model = SentenceTransformer(model_name)

sum_col = "Summary"
art_col = "Article"
# sum_col = "Sentence in Summary"
# art_col = "Best Match Sentences From Article"

# Handle NaNs
df[sum_col] = df[sum_col].fillna("").astype(str)
df[art_col] = df[art_col].fillna("").astype(str)

# # Function for cosine similarity
# def cosine_sim(v1, v2):
#     v1 = np.array(v1)
#     v2 = np.array(v2)
#     return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# # 4) Batch-encode for speed
# summary_embs = model.encode(df[sum_col].tolist(), batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
# article_embs = model.encode(df[art_col].tolist(), batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

# Tokenizer for the embedding model
enc = tiktoken.encoding_for_model("text-embedding-3-small")
MAX_TOKENS = 8190

def truncate_to_limit(text, max_tokens=MAX_TOKENS):
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        # Truncate to the limit (you can choose a smaller cutoff like 8000 for safety)
        tokens = tokens[:max_tokens]
        text = enc.decode(tokens)
    return text

def get_embeddings_batched(texts, model="text-embedding-3-small", batch_size=50):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [truncate_to_limit(t) for t in texts[i : i + batch_size]]
        response = client.embeddings.create(input=batch, model=model)
        batch_embs = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embs)
        print(f"Processed batch {i // batch_size + 1}/{len(texts) // batch_size + 1}")
    return np.array(all_embeddings)


# # --- Get embeddings ---
# sum_embs = get_embeddings_batched(df[sum_col].tolist(), model="text-embedding-3-small", batch_size=10)
# art_embs = get_embeddings_batched(df[art_col].tolist(), model="text-embedding-3-small", batch_size=10)

# # save to file
# np.save("./tests/output/summary_embeddings.npy", sum_embs)
# np.save("./tests/output/article_embeddings.npy", art_embs)

# np.save("./tests/output/summary_sentence_embeddings.npy", sum_embs)
# np.save("./tests/output/article_sentence_embeddings.npy", art_embs)

# load from file
sum_embs = np.load("./tests/output/summary_embeddings.npy")
art_embs = np.load("./tests/output/article_embeddings.npy")

# sum_embs = np.load("./tests/output/summary_sentence_embeddings.npy")
# art_embs = np.load("./tests/output/article_sentence_embeddings.npy")

# # Batch embedding
# sum_embeddings = client.embeddings.create(
#     input=df[sum_col].tolist(),
#     model="text-embedding-3-small"
# ).data
# art_embeddings = client.embeddings.create(
#     input=df[art_col].tolist(),
#     model="text-embedding-3-small"
# ).data

# # Convert to numpy arrays
# sum_embs = np.array([e.embedding for e in sum_embeddings])
# art_embs = np.array([e.embedding for e in art_embeddings])

# Compute cosine similarity per row
sum_embs = sum_embs / np.linalg.norm(sum_embs, axis=1, keepdims=True)
art_embs = art_embs / np.linalg.norm(art_embs, axis=1, keepdims=True)
sims = np.sum(sum_embs * art_embs, axis=1)

# 5) Cosine similarity per row (since we normalized, cosine = dot product)
# sims = np.sum(summary_embs * article_embs, axis=1)  
df["EMB_SIM"] = sims


# agg_cols = ["score", "EMB_SIM"]
# df_article = df.groupby(["Article"], as_index=False)[agg_cols].mean()

# 7) Correlations (coefficients + p-values)
cols = ["score", "EMB_SIM"]

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

# print("Pearson:\n", corr_table_with_pvals(df_article, cols, "pearson"), "\n")
# print("Spearman:\n", corr_table_with_pvals(df_article, cols, "spearman"), "\n")
# print("Kendall:\n", corr_table_with_pvals(df_article, cols, "kendall"), "\n")

print("Pearson:\n", corr_table_with_pvals(df, cols, "pearson"), "\n")
print("Spearman:\n", corr_table_with_pvals(df, cols, "spearman"), "\n")
print("Kendall:\n", corr_table_with_pvals(df, cols, "kendall"), "\n")

# 8) Save for later
# df.to_csv("./tests/output/dataset_with_emb_sims_sent.csv", index=False)
df.to_csv("./tests/output/dataset_with_emb_sims_article_.csv", index=False)
