# Technology Stack

## Core Technologies
- **Python**: Primary language for data processing and pipeline development
- **Pandas**: Data manipulation and analysis
- **Apache Spark**: Large-scale data processing (PySpark)
- **SQL**: Database queries and data transformations
- **Docker**: Containerization for consistent environments

## Data Storage & Databases
- **PostgreSQL/MySQL**: Relational database storage
- **Apache Parquet**: Columnar storage format
- **CSV/JSON**: Common data interchange formats

## Workflow & Orchestration
- **Apache Airflow**: Workflow orchestration and scheduling
- **Jupyter Notebooks**: Interactive development and analysis
- **Git**: Version control for code and configurations

## AI/ML Libraries
- **scikit-learn**: Machine learning algorithms
- **TensorFlow/PyTorch**: Deep learning frameworks
- **MLflow**: ML experiment tracking and model management

## Common Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Processing
```bash
# Run ETL pipeline
python src/pipeline/etl_main.py

# Execute data quality checks
python src/quality/data_validation.py

# Start Jupyter notebook
jupyter notebook
```

### Testing
```bash
# Run unit tests
pytest tests/

# Run data quality tests
pytest tests/data_quality/
```