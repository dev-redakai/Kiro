"""Setup validation utilities to verify project structure and configuration."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from .config import config_manager
from .logger import get_logger


class SetupValidator:
    """Validates the project setup and configuration."""
    
    def __init__(self):
        """Initialize the setup validator."""
        self.logger = get_logger("setup_validator")
        self.errors = []
        self.warnings = []
    
    def validate_directory_structure(self) -> bool:
        """Validate that all required directories exist."""
        required_dirs = [
            "src",
            "src/pipeline",
            "src/data", 
            "src/quality",
            "src/utils",
            "data",
            "data/raw",
            "data/processed",
            "data/output",
            "data/temp",
            "config",
            "tests",
            "logs"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            self.errors.extend([f"Missing directory: {d}" for d in missing_dirs])
            return False
        
        self.logger.info("Directory structure validation passed")
        return True
    
    def validate_configuration(self) -> bool:
        """Validate the configuration setup."""
        try:
            config = config_manager.config
            config_errors = config.validate()
            
            if config_errors:
                self.errors.extend([f"Configuration error: {e}" for e in config_errors])
                return False
            
            # Check if config file exists
            if not Path("config/pipeline_config.yaml").exists():
                self.warnings.append("Configuration file not found, using defaults")
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Configuration validation failed: {str(e)}")
            return False
    
    def validate_dependencies(self) -> bool:
        """Validate that required dependencies are available."""
        required_packages = [
            "pandas",
            "numpy", 
            "yaml",
            "pathlib"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.errors.extend([f"Missing package: {p}" for p in missing_packages])
            return False
        
        self.logger.info("Dependencies validation passed")
        return True
    
    def validate_logging_setup(self) -> bool:
        """Validate that logging is properly configured."""
        try:
            logger = get_logger("test_logger")
            logger.info("Testing logging setup")
            
            # Check if logs directory exists
            if not Path("logs").exists():
                self.errors.append("Logs directory not found")
                return False
            
            self.logger.info("Logging setup validation passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Logging setup validation failed: {str(e)}")
            return False
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run all validation checks and return results."""
        self.logger.info("Starting full project setup validation")
        
        results = {
            "directory_structure": self.validate_directory_structure(),
            "configuration": self.validate_configuration(),
            "dependencies": self.validate_dependencies(),
            "logging_setup": self.validate_logging_setup()
        }
        
        all_passed = all(results.values())
        
        validation_result = {
            "success": all_passed,
            "results": results,
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        if all_passed:
            self.logger.info("All validation checks passed successfully")
        else:
            self.logger.error(f"Validation failed with {len(self.errors)} errors")
            for error in self.errors:
                self.logger.error(f"  - {error}")
        
        if self.warnings:
            self.logger.warning(f"Found {len(self.warnings)} warnings")
            for warning in self.warnings:
                self.logger.warning(f"  - {warning}")
        
        return validation_result
    
    def create_missing_directories(self) -> None:
        """Create any missing directories."""
        required_dirs = [
            "src/pipeline",
            "src/data", 
            "src/quality",
            "src/utils",
            "data/raw",
            "data/processed",
            "data/output",
            "data/temp",
            "config",
            "tests",
            "logs"
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Created missing directories")


def validate_setup() -> bool:
    """Convenience function to validate the entire setup."""
    validator = SetupValidator()
    result = validator.run_full_validation()
    return result["success"]


if __name__ == "__main__":
    """Run validation when script is executed directly."""
    success = validate_setup()
    sys.exit(0 if success else 1)