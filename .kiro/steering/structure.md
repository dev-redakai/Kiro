# Project Structure

## Directory Organization

```
├── src/                    # Source code
│   ├── pipeline/          # ETL pipeline modules
│   ├── data/              # Data processing utilities
│   ├── quality/           # Data quality and validation
│   ├── models/            # ML models and training scripts
│   └── utils/             # Common utilities and helpers
├── data/                  # Data storage
│   ├── raw/               # Raw input data
│   ├── processed/         # Cleaned and transformed data
│   ├── output/            # Final processed outputs
│   └── temp/              # Temporary processing files
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit and integration tests
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── data_quality/      # Data quality tests
├── config/                # Configuration files
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── docker-compose.yml    # Container orchestration
```

## Code Organization Principles

### Module Structure
- Each module should have a clear, single responsibility
- Use `__init__.py` files to define module interfaces
- Keep related functionality grouped together

### File Naming Conventions
- Use snake_case for Python files and directories
- Prefix test files with `test_`
- Use descriptive names that indicate functionality

### Data File Organization
- Raw data should never be modified in place
- Processed data should include timestamps or version identifiers
- Use consistent file naming patterns: `{dataset}_{processing_stage}_{timestamp}.{format}`

### Configuration Management
- Store environment-specific configs in `config/` directory
- Use YAML or JSON for configuration files
- Never commit sensitive credentials to version control

## Import Conventions
- Use absolute imports from project root
- Group imports: standard library, third-party, local modules
- Use meaningful aliases for commonly used modules (e.g., `import pandas as pd`)