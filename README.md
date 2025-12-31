# Hebrew-LLM-Eval

A comprehensive toolkit for generating data, training, and evaluating benchmark models to assess Large Language Models (LLMs) in Hebrew. This repository focuses on evaluating sequence coherence using transformer-based models.

## Description

Hebrew-LLM-Eval provides tools to:
- Train coherence models that distinguish between coherent and shuffled Hebrew text
- Evaluate trained models on test datasets
- Generate evaluation metrics including accuracy, MRR, Recall@K, and NDCG@K
- Support multiple training strategies (standard, weighted loss, focal loss)
- Enable cross-validation and different data splitting strategies

## Installation

### Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/Hebrew-LLM-Eval.git
cd Hebrew-LLM-Eval

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/Hebrew-LLM-Eval.git
cd Hebrew-LLM-Eval

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate  # On Windows

# Install the package in editable mode
pip install -e .
```

## Usage

The main CLI tool is `eich` (Hebrew for "how"). After installation, you can access it from anywhere in your terminal.

### Training a Coherence Model

Train a model to evaluate sequence coherence in Hebrew text:

```bash
eich coherence train \
  --data-path /path/to/training_data.jsonl \
  --model_name dicta-il/alephbertgimmel-base \
  --output-dir ./results/my_model \
  --epochs 3 \
  --batch-size 32 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --k-max 20
```

**Key Arguments:**
- `--data-path`: Path to training data file (JSONL format)
- `--model_name`: Base model from Hugging Face (default: dicta-il/alephbertgimmel-base)
- `--output-dir`: Directory to save checkpoints and final model
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size per device
- `--learning-rate`: Initial learning rate
- `--max-length`: Maximum sequence length for tokenization
- `--k-max`: Maximum number of shuffled negative examples per text

**Advanced Training Options:**

```bash
# Cross-validation with group-based splitting
eich coherence train \
  --data-path data.jsonl \
  --cv 5 \
  --split-type KEY \
  --split-key source \
  --output-dir ./results/cv_model

# Using weighted loss
eich coherence train \
  --data-path data.jsonl \
  --trainer-type WEIGHTED \
  --class-weights 1.0 20.0 \
  --output-dir ./results/weighted_model

# Using focal loss
eich coherence train \
  --data-path data.jsonl \
  --trainer-type FOCAL \
  --focal-loss-alpha 0.25 0.75 \
  --focal-loss-gamma 2.0 \
  --output-dir ./results/focal_model
```

### Evaluating a Trained Model

Evaluate a trained model on test data:

```bash
eich coherence evaluate \
  --data-path /path/to/test_data.csv \
  --model-path ./results/my_model \
  --output-file ./eval_results.json \
  --batch-size 16 \
  --device cuda \
  --max-length 512 \
  --k-max 20
```

**Key Arguments:**
- `--data-path`: Path to evaluation data file (CSV format)
- `--model-path`: Path to the trained model directory
- `--output-file`: File to save evaluation results (JSON)
- `--batch-size`: Batch size for evaluation
- `--device`: Device for evaluation (cuda or cpu)

### Running the Standalone Evaluation Script

For a fully standalone evaluation experience, you can use the script in the `scripts/` folder:

#### Using UV (Recommended)

The script includes inline metadata for UV, making it self-contained:

```bash
uv run scripts/eval_coherence.py \
  --data-path /path/to/test_data.csv \
  --model-name ./results/my_model \
  --output-path ./results/ \
  --device cuda \
  --num-labels 2 \
  --max-length 512 \
  --k-max 20
```

#### Using Python Directly

First install the required dependencies:

```bash
pip install transformers torch tqdm pandas numpy nltk
```

Then run the script:

```bash
python scripts/eval_coherence.py \
  --data-path /path/to/test_data.csv \
  --model-name ./results/my_model \
  --output-path ./results/ \
  --device cuda \
  --num-labels 2 \
  --max-length 512 \
  --k-max 20
```

**Output:**
- `results/results.json`: Detailed evaluation metrics
- `results/test_data_predicted.csv`: Test data with prediction scores

## Project Structure

```
Hebrew-LLM-Eval/
├── src/
│   └── hebrew_llm_eval/
│       ├── cli.py                    # Main CLI entry point
│       ├── coherence/
│       │   ├── train/                # Training logic
│       │   ├── eval/                 # Evaluation logic
│       │   └── data/                 # Dataset handling
│       └── common/                   # Shared utilities
├── scripts/
│   └── eval_coherence.py             # Standalone evaluation script
├── tests/                            # Unit tests
├── pyproject.toml                    # Project configuration
└── README.md                         # This file
```

## Development

To contribute or modify the code:

```bash
# Install with development dependencies
uv sync --group dev

# Run tests
pytest

# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/
```

## License

See [LICENSE](LICENSE) for details.
