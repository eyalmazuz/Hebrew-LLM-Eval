import gradio as gr
import pandas as pd
import os
import json
import numpy as np

# Load your CSV data
df = pd.read_csv("./Data/output/summarization-7-heb.jsonl_results_20250612_115133.csv")

# Convert article match string to list of tuples
df["Best Match Sentences From Article"] = df["Best Match Sentences From Article"].apply(
    lambda x: eval(x) if isinstance(x, str) else []
)

stances = ["תומך", "ניטרלי", "מתנגד"]
annotations = []

def label_sentence(
    topic_sum, stance_sum, topic_art, stance_art, idx
):
    if all([topic_sum.strip(), stance_sum, topic_art.strip(), stance_art]):
        annotations.append({
            "index": idx,
            "summary_sentence": df.loc[idx, "Sentence in Summary"],
            "article_sentence": df.loc[idx, "Best Match Sentences From Article"],
            "summary_topic": topic_sum.strip(),
            "summary_stance": stance_sum,
            "article_topic": topic_art.strip(),
            "article_stance": stance_art
        })

    next_idx = idx + 1
    if next_idx >= len(df):
        return gr.update(visible=False), gr.update(visible=True), "🎉 All sentences labeled!", "", "", "", "", next_idx
    else:
        sentence = f"{next_idx+1}. {df.loc[next_idx, 'Sentence in Summary']}"
        match = df.loc[next_idx, "Best Match Sentences From Article"]
        match_sentence = match[0][0] if match else "(No match)"
        return sentence, match_sentence, "", "", "", "", next_idx

def save_annotations():
    output_file = "./scripts/labeled/annotations.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    return f"✅ Saved {len(annotations)} annotations to {output_file}"

with gr.Blocks() as demo:
    gr.Markdown("## 🏷️ Dual Stance + Topic Labeling for Summary and Article")

    sentence_box = gr.Textbox(label="Summary Sentence to Label", interactive=False)
    match_box = gr.Textbox(label="Matched Article Sentence", interactive=False)

    gr.Markdown("### 🟨 Summary Sentence Annotation")
    topic_summary = gr.Textbox(label="Topic (Summary Sentence)")
    stance_summary = gr.Dropdown(label="Stance (Summary Sentence)", choices=stances)

    gr.Markdown("### 🟧 Article Sentence Annotation")
    topic_article = gr.Textbox(label="Topic (Article Sentence)")
    stance_article = gr.Dropdown(label="Stance (Article Sentence)", choices=stances)

    idx_state = gr.State(0)

    with gr.Row():
        submit_btn = gr.Button("✅ Submit & Next")
        save_btn = gr.Button("💾 Save Annotations", visible=False)

    status = gr.Textbox(label="Status", interactive=False)

    submit_btn.click(
        fn=label_sentence,
        inputs=[topic_summary, stance_summary, topic_article, stance_article, idx_state],
        outputs=[
            sentence_box, match_box,
            topic_summary, stance_summary,
            topic_article, stance_article,
            idx_state
        ]
    )

    save_btn.click(fn=save_annotations, outputs=status)

    demo.load(
        fn=lambda: (
            f"1. {df.loc[0, 'Sentence in Summary']}",
            df.loc[0, "Best Match Sentences From Article"][0][0] if df.loc[0, "Best Match Sentences From Article"] else "(No match)",
            "", "", "", "", 0
        ),
        outputs=[sentence_box, match_box, topic_summary, stance_summary, topic_article, stance_article, idx_state]
    )

demo.launch()