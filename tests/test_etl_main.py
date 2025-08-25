#!/usr/bin/env python3
"""
Test script for ETLMain orchestrator.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.pipeline.etl_main import ETLMain, create_cli_parser
    print("✓ Successfully imported ETLMain")
    
    # Test initialization
    etl = ETLMain()
    print("✓ Successfully initialized ETLMain")
    
    # Test configuration validation
    is_valid = etl.validate_configuration()
    print(f"✓ Configuration validation: {'Valid' if is_valid else 'Invalid'}")
    
    # Test CLI parser
    parser = create_cli_parser()
    print("✓ Successfully created CLI parser")
    
    # Test status
    status = etl.get_status()
    print(f"✓ Pipeline status: {status['current_stage']}")
    
    print("\nAll basic tests passed!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)