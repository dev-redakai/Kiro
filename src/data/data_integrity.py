"""
Data Integrity Validation Module

This module provides functionality for validating data integrity of exported files.
"""

import pandas as pd
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from ..utils.logger import get_logger


class DataIntegrityValidator:
    """
    Validates data integrity for exported files.
    """
    
    def __init__(self):
        """Initialize the DataIntegrityValidator."""
        self.logger = get_logger(__name__)
    
    def calculate_file_checksum(self, file_path: str) -> str:
        """
        Calculate MD5 checksum of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 checksum as hex string
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            raise
    
    def validate_data_consistency(self, original_data: pd.DataFrame,
                                 exported_file_path: str,
                                 file_format: str = 'csv') -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that exported data matches original data.
        
        Args:
            original_data: Original DataFrame
            exported_file_path: Path to exported file
            file_format: Format of exported file ('csv' or 'parquet')
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        try:
            # Load exported data
            if file_format.lower() == 'csv':
                exported_data = pd.read_csv(exported_file_path)
            elif file_format.lower() == 'parquet':
                exported_data = pd.read_parquet(exported_file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Validation checks
            validation_report = {
                "file_path": exported_file_path,
                "file_format": file_format,
                "validation_timestamp": pd.Timestamp.now().isoformat(),
                "checks": {}
            }
            
            # Check shape
            shape_match = original_data.shape == exported_data.shape
            validation_report["checks"]["shape_match"] = {
                "passed": shape_match,
                "original_shape": original_data.shape,
                "exported_shape": exported_data.shape
            }
            
            # Check columns
            columns_match = list(original_data.columns) == list(exported_data.columns)
            validation_report["checks"]["columns_match"] = {
                "passed": columns_match,
                "original_columns": list(original_data.columns),
                "exported_columns": list(exported_data.columns)
            }
            
            # Check data types (allowing for some flexibility)
            dtype_issues = []
            for col in original_data.columns:
                if col in exported_data.columns:
                    orig_dtype = str(original_data[col].dtype)
                    exp_dtype = str(exported_data[col].dtype)
                    if orig_dtype != exp_dtype:
                        # Allow some common conversions
                        if not self._is_acceptable_dtype_conversion(orig_dtype, exp_dtype):
                            dtype_issues.append({
                                "column": col,
                                "original_dtype": orig_dtype,
                                "exported_dtype": exp_dtype
                            })
            
            validation_report["checks"]["dtype_consistency"] = {
                "passed": len(dtype_issues) == 0,
                "issues": dtype_issues
            }
            
            # Check for data equality (sample-based for large datasets)
            if len(original_data) > 10000:
                # Sample-based comparison for large datasets
                sample_size = min(1000, len(original_data))
                sample_indices = original_data.sample(n=sample_size).index
                orig_sample = original_data.loc[sample_indices].reset_index(drop=True)
                exp_sample = exported_data.loc[sample_indices].reset_index(drop=True)
                data_match = orig_sample.equals(exp_sample)
            else:
                # Full comparison for smaller datasets
                data_match = original_data.reset_index(drop=True).equals(
                    exported_data.reset_index(drop=True)
                )
            
            validation_report["checks"]["data_equality"] = {
                "passed": data_match,
                "comparison_method": "sample" if len(original_data) > 10000 else "full"
            }
            
            # Calculate file checksum
            file_checksum = self.calculate_file_checksum(exported_file_path)
            validation_report["file_checksum"] = file_checksum
            
            # Overall validation result
            all_checks_passed = all(
                check["passed"] for check in validation_report["checks"].values()
            )
            
            validation_report["overall_valid"] = all_checks_passed
            
            if all_checks_passed:
                self.logger.info(f"Data integrity validation passed for {exported_file_path}")
            else:
                self.logger.warning(f"Data integrity validation failed for {exported_file_path}")
            
            return all_checks_passed, validation_report
            
        except Exception as e:
            self.logger.error(f"Data integrity validation failed: {e}")
            validation_report = {
                "file_path": exported_file_path,
                "validation_timestamp": pd.Timestamp.now().isoformat(),
                "error": str(e),
                "overall_valid": False
            }
            return False, validation_report
    
    def _is_acceptable_dtype_conversion(self, original_dtype: str, exported_dtype: str) -> bool:
        """
        Check if a dtype conversion is acceptable during export/import.
        
        Args:
            original_dtype: Original data type
            exported_dtype: Exported data type
            
        Returns:
            True if conversion is acceptable
        """
        # Common acceptable conversions
        acceptable_conversions = {
            ('int64', 'int32'),
            ('int32', 'int64'),
            ('float64', 'float32'),
            ('float32', 'float64'),
            ('object', 'string'),
            ('string', 'object'),
            ('datetime64[ns]', 'object'),  # CSV export often converts datetime to string
            ('object', 'datetime64[ns]')   # Parquet might preserve datetime
        }
        
        return (original_dtype, exported_dtype) in acceptable_conversions
    
    def validate_exported_files(self, export_paths: Dict[str, str],
                               original_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple exported files.
        
        Args:
            export_paths: Dictionary mapping format to file path
            original_data: Original DataFrame
            
        Returns:
            Dictionary mapping format to validation report
        """
        validation_results = {}
        
        for file_format, file_path in export_paths.items():
            if Path(file_path).exists():
                is_valid, report = self.validate_data_consistency(
                    original_data, file_path, file_format
                )
                validation_results[file_format] = report
            else:
                validation_results[file_format] = {
                    "file_path": file_path,
                    "error": "File does not exist",
                    "overall_valid": False
                }
        
        return validation_results
    
    def save_validation_report(self, validation_report: Dict[str, Any],
                              output_path: str):
        """
        Save validation report to file.
        
        Args:
            validation_report: Validation report dictionary
            output_path: Path to save the report
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)
            self.logger.info(f"Validation report saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")
            raise