import json
from datasets import Dataset, load_dataset


def preprocess_hesum(dataset_name):
    """Preprocess the biunlp/HeSum dataset."""
    dataset = load_dataset(dataset_name)
    # Get the 'train' split which contains the data
    if 'train' in dataset:
        dataset = dataset['train']
    return dataset.filter(lambda x: x["article"] and x["summary"])


def preprocess_custom(file_path):
    """Preprocess custom dataset containing 'text_raw'."""
    articles = []
    summaries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            source_text = record.get('text_raw', "")
            summary_text = record.get('summary', "")
            articles.append(source_text)
            summaries.append(summary_text)

    return Dataset.from_dict({"article": articles, "summary": summaries})


def load_data(dataset_name, dataset_path=None):
    """Load and preprocess dataset based on the type."""
    if dataset_name == "biunlp/HeSum":
        dataset = preprocess_hesum(dataset_name)
    elif dataset_name == "custom" and dataset_path:
        dataset = preprocess_custom(dataset_path)
    else:
        raise ValueError("Invalid dataset name or missing dataset_path for custom data.")
    return dataset
