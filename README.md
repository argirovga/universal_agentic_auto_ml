# Multi-Agent ML System

A multi-agent system built with LangGraph for automatic regression on tabular dataset (Kaggle format).

## Architecture

```
START -> Coordinator -> Explorer (EDA) -> Engineer (modeling) -> Critic (evaluation)
                                               ^                      |
                                               +---- if not good -----+
                                                     (max 5 iterations)
                                         if good -> Submission -> END
```

### Agents

| Agent | Role | Key Actions |
|-------|------|-------------|
| **Coordinator** | Initializes pipeline state, describes the task | Sets paths, config |
| **Explorer** | Exploratory Data Analysis | Data profiling, correlations, feature suggestions |
| **Engineer** | Model training & feature engineering | Selects model, tunes hyperparameters, trains |
| **Critic** | Results evaluation & decision-making | Evaluates metrics, decides to iterate or submit |

### Components

- **LLM Backend**: Dual support — Ollama (local) + HuggingFace Inference API
- **RAG**: ChromaDB + sentence-transformers knowledge base (ML best practices)
- **Tools**: Data profiling, ML training/prediction, input validation
- **Memory**: JSON-based experiment tracking
- **Benchmarking**: Agent performance metrics + ML metrics reports
- **Safety**: Input validation, output clipping, timeout guards, error recovery

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings (HF_TOKEN, LLM_PROVIDER, etc.)
```

### Prerequisites

- Python 3.12+
- Ollama installed with a model (e.g., `ollama pull qwen3-coder:30b`)
- OR HuggingFace API token for cloud inference

## Usage

```bash
# Run with default settings (Ollama)
python main.py

# Run with HuggingFace backend
python main.py --provider huggingface

# Limit iterations
python main.py --max-iterations 3

# Override model
python main.py --model mistral
```

## Project Structure

```
llm_auto_ml_competition/
├── config/settings.py           # Configuration (paths, LLM, ML params)
├── data/raw_data/               # Train/test datasets
├── knowledge_base/
│   ├── documents/               # RAG source documents (MD)
│   └── chroma_db/               # ChromaDB vector store
├── src/
│   ├── agents/                  # Coordinator, Explorer, Engineer, Critic
│   ├── llm/                     # LLM provider abstraction (Ollama, HF)
│   ├── rag/                     # Document indexer + retriever
│   ├── tools/                   # Data tools, ML tools, validation
│   ├── memory/                  # Experiment tracking
│   ├── benchmark/               # Performance evaluation
│   └── graph.py                 # LangGraph workflow definition
├── outputs/                     # Submissions, reports, logs
├── tests/                       # Unit tests
├── main.py                      # Entry point
└── requirements.txt
```

## Outputs

After running the pipeline, check `outputs/`:
- `submission.csv` — Kaggle submission file
- `experiments.json` — Full experiment history
- `benchmark_report.md` — Agent and model performance report
- `run.log` — Detailed execution log

## Testing

```bash
python -m pytest tests/ -v
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph |
| LLM (local) | Ollama (qwen3-coder) |
| LLM (cloud) | HuggingFace Inference API |
| RAG | ChromaDB + sentence-transformers |
| ML | scikit-learn, LightGBM, XGBoost |
| Data | pandas |
| Config | pydantic-settings, python-dotenv |
