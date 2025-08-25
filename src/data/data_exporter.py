"""
Data Export Module

This module provides functionality for exporting processed data in various formats
including CSV and Parquet, with versioning and timestamp management.
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import hashlib
import logging

from ..utils.config import PipelineConfig
from ..utils.logger import get_logger


class DataExporter:
    """
    Handles exporting processed data in multiple formats with versioning and metadata.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the DataExporter.
        
        Args:
            config: Configuration object containing export settings
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.output_path = Path(config.output_data_path)
        self.processed_path = Path(config.processed_data_path)
        
        # Ensure output directories exist
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary output directories."""
        directories = [
            self.output_path,
            self.output_path / "csv",
            self.output_path / "parquet",
            self.output_path / "analytical",
            self.output_path / "metadata",
            self.processed_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def _generate_timestamp(self) -> str:
        """Generate timestamp string for file versioning."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _generate_file_hash(self, data: pd.DataFrame) -> str:
        """Generate hash for data integrity validation."""
        # Create a hash based on data shape and sample values
        data_info = f"{data.shape}_{data.dtypes.to_string()}_{data.head().to_string()}"
        return hashlib.md5(data_info.encode()).hexdigest()[:8]
    
    def _create_metadata(self, data: pd.DataFrame, export_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for exported data."""
        return {
            "export_timestamp": datetime.now().isoformat(),
            "data_shape": list(data.shape),
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "file_hash": self._generate_file_hash(data),
            "export_info": export_info,
            "memory_usage": int(data.memory_usage(deep=True).sum()),
            "null_counts": {col: int(count) for col, count in data.isnull().sum().items()}
        }
    
    def export_csv(self, data: pd.DataFrame, filename: str, 
                   include_timestamp: bool = True, **kwargs) -> str:
        """
        Export data to CSV format.
        
        Args:
            data: DataFrame to export
            filename: Base filename (without extension)
            include_timestamp: Whether to include timestamp in filename
            **kwargs: Additional arguments for pandas.to_csv()
            
        Returns:
            Path to exported file
        """
        try:
            timestamp = self._generate_timestamp() if include_timestamp else ""
            file_suffix = f"_{timestamp}" if timestamp else ""
            csv_filename = f"{filename}{file_suffix}.csv"
            csv_path = self.output_path / "csv" / csv_filename
            
            # Default CSV export parameters
            csv_params = {
                'index': False,
                'encoding': 'utf-8',
                'date_format': '%Y-%m-%d %H:%M:%S'
            }
            csv_params.update(kwargs)
            
            # Export to CSV
            data.to_csv(csv_path, **csv_params)
            
            # Create metadata
            metadata = self._create_metadata(data, {
                "format": "csv",
                "filename": csv_filename,
                "export_params": csv_params
            })
            
            # Save metadata
            metadata_path = self.output_path / "metadata" / f"{filename}{file_suffix}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Successfully exported CSV: {csv_path}")
            return str(csv_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV {filename}: {str(e)}")
            raise
    
    def export_parquet(self, data: pd.DataFrame, filename: str,
                      include_timestamp: bool = True, **kwargs) -> str:
        """
        Export data to Parquet format for efficient storage.
        
        Args:
            data: DataFrame to export
            filename: Base filename (without extension)
            include_timestamp: Whether to include timestamp in filename
            **kwargs: Additional arguments for pandas.to_parquet()
            
        Returns:
            Path to exported file
        """
        try:
            timestamp = self._generate_timestamp() if include_timestamp else ""
            file_suffix = f"_{timestamp}" if timestamp else ""
            parquet_filename = f"{filename}{file_suffix}.parquet"
            parquet_path = self.output_path / "parquet" / parquet_filename
            
            # Default Parquet export parameters
            parquet_params = {
                'engine': 'pyarrow',
                'compression': 'snappy',
                'index': False
            }
            parquet_params.update(kwargs)
            
            # Export to Parquet
            data.to_parquet(parquet_path, **parquet_params)
            
            # Create metadata
            metadata = self._create_metadata(data, {
                "format": "parquet",
                "filename": parquet_filename,
                "export_params": parquet_params
            })
            
            # Save metadata
            metadata_path = self.output_path / "metadata" / f"{filename}{file_suffix}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Successfully exported Parquet: {parquet_path}")
            return str(parquet_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export Parquet {filename}: {str(e)}")
            raise
    
    def export_analytical_tables(self, analytical_data: Dict[str, pd.DataFrame],
                                include_timestamp: bool = True) -> Dict[str, Dict[str, str]]:
        """
        Export analytical tables in both CSV and Parquet formats.
        
        Args:
            analytical_data: Dictionary of table_name -> DataFrame
            include_timestamp: Whether to include timestamp in filenames
            
        Returns:
            Dictionary mapping table names to export paths
        """
        export_results = {}
        
        try:
            for table_name, data in analytical_data.items():
                self.logger.info(f"Exporting analytical table: {table_name}")
                
                # Export in both formats
                csv_path = self.export_csv(
                    data, 
                    f"analytical_{table_name}",
                    include_timestamp=include_timestamp
                )
                
                parquet_path = self.export_parquet(
                    data,
                    f"analytical_{table_name}",
                    include_timestamp=include_timestamp
                )
                
                export_results[table_name] = {
                    'csv': csv_path,
                    'parquet': parquet_path
                }
            
            self.logger.info(f"Successfully exported {len(analytical_data)} analytical tables")
            return export_results
            
        except Exception as e:
            self.logger.error(f"Failed to export analytical tables: {str(e)}")
            raise
    
    def export_processed_data(self, data: pd.DataFrame, dataset_name: str = "processed_data",
                             formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Export processed data in specified formats.
        
        Args:
            data: Processed DataFrame to export
            dataset_name: Name for the dataset
            formats: List of formats to export ('csv', 'parquet'). If None, uses config.
            
        Returns:
            Dictionary mapping format to file path
        """
        if formats is None:
            formats = self.config.output_formats
        
        export_paths = {}
        
        try:
            for format_type in formats:
                if format_type.lower() == 'csv':
                    export_paths['csv'] = self.export_csv(data, dataset_name)
                elif format_type.lower() == 'parquet':
                    export_paths['parquet'] = self.export_parquet(data, dataset_name)
                else:
                    self.logger.warning(f"Unsupported format: {format_type}")
            
            return export_paths
            
        except Exception as e:
            self.logger.error(f"Failed to export processed data: {str(e)}")
            raise


class DataVersionManager:
    """
    Manages data versioning and tracks export history.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the DataVersionManager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.version_file = Path(config.output_data_path) / "version_history.json"
        self.version_history = self._load_version_history()
    
    def _load_version_history(self) -> Dict[str, Any]:
        """Load version history from file."""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load version history: {e}")
        
        return {"versions": [], "current_version": None}
    
    def _save_version_history(self):
        """Save version history to file."""
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self.version_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save version history: {e}")
    
    def create_version(self, dataset_name: str, export_paths: Dict[str, str],
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new version entry for exported data.
        
        Args:
            dataset_name: Name of the dataset
            export_paths: Dictionary of format -> file path
            metadata: Additional metadata for the version
            
        Returns:
            Version identifier
        """
        version_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        version_entry = {
            "version_id": version_id,
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "export_paths": export_paths,
            "metadata": metadata or {}
        }
        
        self.version_history["versions"].append(version_entry)
        self.version_history["current_version"] = version_id
        
        self._save_version_history()
        
        self.logger.info(f"Created version: {version_id}")
        return version_id
    
    def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific version."""
        for version in self.version_history["versions"]:
            if version["version_id"] == version_id:
                return version
        return None
    
    def list_versions(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all versions, optionally filtered by dataset name."""
        versions = self.version_history["versions"]
        if dataset_name:
            versions = [v for v in versions if v["dataset_name"] == dataset_name]
        return sorted(versions, key=lambda x: x["timestamp"], reverse=True)
    
    def get_latest_version(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest version for a specific dataset."""
        dataset_versions = self.list_versions(dataset_name)
        return dataset_versions[0] if dataset_versions else None


class ExportManager:
    """
    High-level manager for coordinating data export operations.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the ExportManager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.exporter = DataExporter(config)
        self.version_manager = DataVersionManager(config)
    
    def export_complete_dataset(self, processed_data: pd.DataFrame,
                               analytical_tables: Dict[str, pd.DataFrame],
                               dataset_name: str = "complete_dataset") -> Dict[str, Any]:
        """
        Export complete dataset including processed data and analytical tables.
        
        Args:
            processed_data: Main processed dataset
            analytical_tables: Dictionary of analytical tables
            dataset_name: Name for the complete dataset
            
        Returns:
            Export summary with paths and version information
        """
        try:
            self.logger.info(f"Starting complete dataset export: {dataset_name}")
            
            # Export processed data
            processed_paths = self.exporter.export_processed_data(
                processed_data, f"{dataset_name}_processed"
            )
            
            # Export analytical tables
            analytical_paths = self.exporter.export_analytical_tables(analytical_tables)
            
            # Combine all export paths
            all_export_paths = {
                "processed_data": processed_paths,
                "analytical_tables": analytical_paths
            }
            
            # Create version entry
            metadata = {
                "processed_records": len(processed_data),
                "analytical_tables": list(analytical_tables.keys()),
                "export_formats": self.config.output_formats
            }
            
            version_id = self.version_manager.create_version(
                dataset_name, all_export_paths, metadata
            )
            
            export_summary = {
                "version_id": version_id,
                "export_paths": all_export_paths,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Successfully completed dataset export: {version_id}")
            return export_summary
            
        except Exception as e:
            self.logger.error(f"Failed to export complete dataset: {str(e)}")
            raise