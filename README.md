# Project Overview
Hebrew-LLM-Stance-Eval is a toolkit for generating data, training, and evaluating benchmark models to assess large language models (LLMs) in Hebrew. The repository provides scripts and utilities for:

- Data preparation and labeling
- Topic and stance detection
- Model training and evaluation
- Benchmarking LLMs on Hebrew datasets

The project supports custom datasets and includes pipelines for matching sentences, detecting topics, and evaluating stance preservation between articles and summaries.

## Installation
1. Clone the repository:
```
git clone https://github.com/yourusername/Hebrew-LLM-Eval.git
cd Hebrew-LLM-Eval
```

2. Create a Python virtual environment (recommended):
```
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```


## Running the Pipeline
The main stance and topic detection pipeline can be run using the `clean_pipeline.py` script. Example usage:
```
python -m scripts.clean_pipeline --data custom --path Data/test_data.csv --output-dir ./scripts/output/test_csv_input.json --save-matches
```

<b> Arguments: </b>
- `--data`: Dataset name (biunlp/HeSum or custom)
- `--path`: Path to your dataset file (CSV or JSON)
- `--output-dir`: Output file for results (JSON)
- `--save-matches`: Save matching results to CSV/JSON
- `--threshold`: (Optional) Matching threshold (default: 0.85)
- `--top-k-matches`: (Optional) Number of top matches to consider (default: 1)
- `--topic_detection_model`: (Optional) Topic detection model (default: dicta-il/dictalm2.0)

<b> Example: </b>
```
python -m scripts.clean_pipeline --data custom --path Data/test_data.csv --output-dir ./scripts/output/results.json --save-matches
```

This will process your dataset, perform sentence matching, topic detection, and stance evaluation, and save the results in the specified output directory.