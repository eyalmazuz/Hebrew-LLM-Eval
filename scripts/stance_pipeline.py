# split article and summary into sentences
# use Me5 as the matching method (sentence matching file)
# use stance detection/sentiment analysis model to classify the sentences that containes stances for both article and summary
# classify the stance for each sentence (favor, neutral, against) for both article and summary 
# check stance presevation for each sentence in the article and summary by calculating the difference (or KL-divergence) between the stances of the sentences in the article and summary

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import pandas as pd

def load_matching_matrix(csv_path):
    """Load the sentence matching matrix from a CSV file."""
    df = pd.read_csv(csv_path)
    print("Matching Matrix Loaded:")
    print(df.head())
    return df

def load_model(model_name):
    """Load the stance detection model and tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def classify_stance(sentences, model, tokenizer):
    """Classify stance (sentiment) for a list of sentences and return labels with probabilities."""
    labels = ['negative', 'neutral', 'positive']  # Adjust according to model output labels
    results = []
    probs = []
    
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()
        predicted_class = torch.argmax(logits, dim=1).item()
        results.append((labels[predicted_class], probabilities[predicted_class]))
        probs.append(probabilities)
    
    return results, probs

def kl_divergence(p, q):
    """Compute KL-divergence between two probability distributions."""
    p = torch.tensor(p)
    q = torch.tensor(q)
    return F.kl_div(q.log(), p, reduction='sum').item()

def compute_stance_preservation(df, model, tokenizer):
    """Compute stance preservation and KL-divergence between sentiment distributions."""
    stance_preservation = []
    # summary_stance_scores = []
    # article_stance_scores = []
    kl_divergences = []
    
    for idx, row in df.iterrows():
        summary_sentence = row['Sentence in Summary']
        article_sentence = row['Best Match Sentence From Article']
        
        summary_results, summary_probs = classify_stance([summary_sentence], model, tokenizer)
        article_results, article_probs = classify_stance([article_sentence], model, tokenizer)

        summary_stance, summary_score = summary_results[0]
        article_stance, article_score = article_results[0]
        
        # summary_stance_scores.append(f"{summary_stance} ({summary_score:.4f})")
        # article_stance_scores.append(f"{article_stance} ({article_score:.4f})")
        
        kl_score = kl_divergence(summary_probs[0], article_probs[0])
        kl_divergences.append(kl_score)
        
        if summary_stance == article_stance:
            stance_preservation.append('Preserved')
        else:
            stance_preservation.append('Not Preserved')
    
    # df['Summary Stance (Score)'] = summary_stance_scores
    # df['Article Stance (Score)'] = article_stance_scores
    df['Stance Preservation'] = stance_preservation
    df['KL Divergence'] = kl_divergences
    return df


if __name__ == '__main__':
    # Load data
    path_to_csv = 'Data/output/me5_summarization-7-heb.jsonl_results_20250305_195549.csv'
    df = load_matching_matrix(path_to_csv)
    
    # Load model and tokenizer
    model_name = 'dicta-il/dictabert-sentiment'
    model, tokenizer = load_model(model_name)
    
    # Classify stance for each sentence in article and summary
    df = compute_stance_preservation(df, model, tokenizer)
    
    # Save results
    output_path = 'Data/output/stance_preservation.csv'
    df.to_csv(output_path, index=False)
    print(f"Stance preservation results saved to {output_path}")
