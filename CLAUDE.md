# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PID Classify is an AI-powered web application that predicts Japanese industrial part number categories (品番カテゴリ) based on part names (品名) and model numbers (型式). Uses machine learning (Multinomial Naive Bayes) to suggest 3-character alphanumeric category codes.

## Development Commands

### Running the Application
```bash
python main.py  # Starts FastAPI server on port 8880
```

### Code Quality
```bash
ruff check .    # Linting
ruff format .   # Formatting  
mypy .          # Type checking
pytest          # Run tests
pytest --cov   # Run tests with coverage
```

### Dependencies
```bash
uv pip compile pyproject.toml --output-file requirements.txt  # Update requirements
pip install -e .[dev]  # Install with dev dependencies
```

### TypeScript Build
```bash
cd static && npx tsc  # Compiles src/main.ts to main.js
```

### Docker
```bash
docker build -t pid_classify .
docker run -v $(pwd)/data:/work/data -p 8880:8880 pid_classify  # Requires data volume
```

## Architecture

### Core ML Pipeline
1. **Data**: SQLite database (`/data/cwz.db`) with 品番, 品名, 型式, カテゴリ columns
2. **Classification**: `pid_classify.py` uses Multinomial Naive Bayes + HashingVectorizer on character trigrams
3. **Category Mapping**: `pid_category.py` manages standard part category definitions from CSV
4. **Web Interface**: FastAPI server (`main.py`) with TypeScript frontend

### Key Components
- **main.py**: FastAPI endpoints for prediction, search, and category browsing
- **pid_classify.py**: ML classifier with >60% accuracy threshold requirement
- **static/src/main.ts**: Frontend with auto-completion and real-time search (400ms debounce)
- **templates/index.html**: Bootstrap 5 UI with probability badges and category exploration

### Data Flow
User input → Exact match check → ML prediction if no match → Probability-ranked suggestions → Category exploration

### ML Model Details
- **Features**: Tab-separated part name + model text
- **Preprocessing**: Character trigram vectorization
- **Training**: Requires SQLite database with historical part classifications
- **Category Extraction**: 3-character prefix from existing part numbers