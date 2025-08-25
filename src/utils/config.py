"""Configuration management system with YAML support."""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class PipelineConfig:
    """Main pipeline configuration class."""
    
    # Data processing settings
    chunk_size: int = 50000
    max_memory_usage: int = 4 * 1024 * 1024 * 1024  # 4GB
    
    # File paths
    input_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    output_data_path: str = "data/output"
    temp_data_path: str = "data/temp"
    
    # Output formats
    output_formats: List[str] = field(default_factory=lambda: ['csv', 'parquet'])
    
    # Anomaly detection settings
    anomaly_threshold: float = 3.0
    anomaly_sensitivity: float = 0.95
    
    # Logging settings
    log_level: str = 'INFO'
    log_file: str = 'pipeline.log'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Data cleaning settings
    standardization_rules: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced validation configuration
    validation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Dashboard settings
    dashboard_port: int = 8501
    dashboard_host: str = 'localhost'
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Create default config file if it doesn't exist
            default_config = cls()
            default_config.save_to_file(config_path)
            return default_config
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if config_data is None:
            config_data = {}
        
        # Create instance with loaded data
        return cls(**config_data)
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass to dict
        config_dict = {
            'chunk_size': self.chunk_size,
            'max_memory_usage': self.max_memory_usage,
            'input_data_path': self.input_data_path,
            'processed_data_path': self.processed_data_path,
            'output_data_path': self.output_data_path,
            'temp_data_path': self.temp_data_path,
            'output_formats': self.output_formats,
            'anomaly_threshold': self.anomaly_threshold,
            'anomaly_sensitivity': self.anomaly_sensitivity,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'log_format': self.log_format,
            'standardization_rules': self.standardization_rules,
            'validation_rules': self.validation_rules,
            'validation_config': self.validation_config,
            'dashboard_port': self.dashboard_port,
            'dashboard_host': self.dashboard_host
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get all data paths as a dictionary."""
        return {
            'input': self.input_data_path,
            'processed': self.processed_data_path,
            'output': self.output_data_path,
            'temp': self.temp_data_path
        }
    
    def validate(self) -> List[str]:
        """Validate configuration settings and return list of errors."""
        errors = []
        
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.max_memory_usage <= 0:
            errors.append("max_memory_usage must be positive")
        
        if self.anomaly_threshold <= 0:
            errors.append("anomaly_threshold must be positive")
        
        if not (0 < self.anomaly_sensitivity <= 1):
            errors.append("anomaly_sensitivity must be between 0 and 1")
        
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}")
        
        valid_formats = ['csv', 'parquet', 'json']
        for fmt in self.output_formats:
            if fmt not in valid_formats:
                errors.append(f"Invalid output format: {fmt}. Valid formats: {valid_formats}")
        
        return errors


class ConfigManager:
    """Configuration manager for the pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or "config/pipeline_config.yaml"
        self._config = None
    
    @property
    def config(self) -> PipelineConfig:
        """Get the current configuration."""
        if self._config is None:
            self._config = PipelineConfig.from_file(self.config_path)
        return self._config
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._config = PipelineConfig.from_file(self.config_path)
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        self.config.save_to_file(self.config_path)
    
    def validate_config(self) -> List[str]:
        """Validate current configuration."""
        return self.config.validate()


# Global configuration instance
config_manager = ConfigManager()