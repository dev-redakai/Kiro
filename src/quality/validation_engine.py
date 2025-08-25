"""
Data validation engine for comprehensive data quality checks.

This module provides the ValidationEngine class that performs data quality
validation, duplicate detection, revenue calculation, and generates
comprehensive validation reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationEngine:
    """
    Comprehensive data validation engine.
    
    This class provides methods for detecting duplicates, calculating revenue,
    validating data quality, and generating detailed validation reports.
    """
    
    def __init__(self):
        """Initialize the ValidationEngine."""
        self.validation_stats = {
            'total_records': 0,
            'valid_records': 0,
            'duplicate_records': 0,
            'validation_errors': {},
            'revenue_calculations': 0
        }
        self.duplicate_order_ids = set()
        self.validation_rules = self._setup_validation_rules()
    
    def _setup_validation_rules(self) -> Dict[str, Dict]:
        """Set up comprehensive validation rules for each field."""
        return {
            'order_id': {
                'required': True,
                'type': str,
                'min_length': 1,
                'max_length': 100,
                'pattern': r'^[A-Za-z0-9\-_]+$'
            },
            'product_name': {
                'required': True,
                'type': str,
                'min_length': 1,
                'max_length': 500
            },
            'category': {
                'required': True,
                'type': str,
                'min_length': 1,
                'max_length': 100
            },
            'quantity': {
                'required': True,
                'type': (int, float),
                'min_value': 1,
                'max_value': 10000
            },
            'unit_price': {
                'required': True,
                'type': (int, float),
                'min_value': 0.01,
                'max_value': 1000000.0
            },
            'discount_percent': {
                'required': True,
                'type': (int, float),
                'min_value': 0.0,
                'max_value': 1.0
            },
            'region': {
                'required': True,
                'type': str,
                'min_length': 1,
                'max_length': 100
            },
            'sale_date': {
                'required': True,
                'type': (datetime, pd.Timestamp)
            },
            'customer_email': {
                'required': False,
                'type': str,
                'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            }
        }
    
    def detect_duplicate_order_ids(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle duplicate order IDs.
        
        Args:
            data: DataFrame containing order data
            
        Returns:
            DataFrame with duplicate information added
        """
        logger.info(f"Detecting duplicate order IDs in {len(data)} records")
        
        if 'order_id' not in data.columns:
            logger.warning("No order_id column found for duplicate detection")
            return data
        
        # Create a copy to avoid modifying original data
        result_data = data.copy()
        
        # Find duplicates
        duplicate_mask = result_data['order_id'].duplicated(keep=False)
        duplicate_count = duplicate_mask.sum()
        
        # Add duplicate flag column
        result_data['is_duplicate'] = duplicate_mask
        
        # Track duplicate order IDs
        duplicate_order_ids = result_data[duplicate_mask]['order_id'].unique()
        self.duplicate_order_ids.update(duplicate_order_ids)
        
        # Update statistics
        self.validation_stats['duplicate_records'] += duplicate_count
        
        logger.info(f"Found {duplicate_count} duplicate records with {len(duplicate_order_ids)} unique order IDs")
        
        return result_data
    
    def calculate_revenue(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate revenue using the formula: quantity * unit_price * (1 - discount_percent).
        
        Args:
            data: DataFrame containing sales data
            
        Returns:
            DataFrame with revenue column added
        """
        logger.info(f"Calculating revenue for {len(data)} records")
        
        required_columns = ['quantity', 'unit_price', 'discount_percent']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns for revenue calculation: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create a copy to avoid modifying original data
        result_data = data.copy()
        
        # Calculate revenue with error handling
        try:
            result_data['revenue'] = (
                result_data['quantity'] * 
                result_data['unit_price'] * 
                (1 - result_data['discount_percent'])
            )
            
            # Handle any NaN or infinite values
            invalid_revenue_mask = ~np.isfinite(result_data['revenue'])
            invalid_count = invalid_revenue_mask.sum()
            
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} records with invalid revenue calculations, setting to 0")
                result_data.loc[invalid_revenue_mask, 'revenue'] = 0.0
            
            # Ensure revenue is non-negative
            negative_revenue_mask = result_data['revenue'] < 0
            negative_count = negative_revenue_mask.sum()
            
            if negative_count > 0:
                logger.warning(f"Found {negative_count} records with negative revenue, setting to 0")
                result_data.loc[negative_revenue_mask, 'revenue'] = 0.0
            
            # Update statistics
            self.validation_stats['revenue_calculations'] += len(data)
            
            logger.info(f"Successfully calculated revenue for {len(data)} records")
            
        except Exception as e:
            logger.error(f"Error calculating revenue: {str(e)}")
            # Set default revenue of 0 if calculation fails
            result_data['revenue'] = 0.0
        
        return result_data
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality validation.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Performing data quality validation on {len(data)} records")
        
        validation_results = {
            'total_records': len(data),
            'valid_records': 0,
            'field_validation_results': {},
            'overall_quality_score': 0.0,
            'validation_errors': []
        }
        
        # Update total records
        self.validation_stats['total_records'] += len(data)
        
        # Validate each field according to rules
        for field_name, rules in self.validation_rules.items():
            if field_name in data.columns:
                field_results = self._validate_field(data[field_name], field_name, rules)
                validation_results['field_validation_results'][field_name] = field_results
            elif rules.get('required', False):
                validation_results['validation_errors'].append(
                    f"Required field '{field_name}' is missing from data"
                )
        
        # Calculate overall quality metrics
        total_field_validations = 0
        total_valid_values = 0
        
        for field_name, field_results in validation_results['field_validation_results'].items():
            total_field_validations += field_results['total_values']
            total_valid_values += field_results['valid_values']
        
        # Calculate quality score
        if total_field_validations > 0:
            validation_results['overall_quality_score'] = total_valid_values / total_field_validations
        
        # Count valid records (records with no validation errors across all fields)
        valid_record_mask = pd.Series([True] * len(data))
        
        for field_name, field_results in validation_results['field_validation_results'].items():
            if field_name in data.columns:
                field_valid_mask = self._get_field_valid_mask(data[field_name], field_name, self.validation_rules[field_name])
                valid_record_mask &= field_valid_mask
        
        validation_results['valid_records'] = valid_record_mask.sum()
        self.validation_stats['valid_records'] += validation_results['valid_records']
        
        logger.info(f"Validation completed. Quality score: {validation_results['overall_quality_score']:.2%}")
        
        return validation_results
    
    def _validate_field(self, field_data: pd.Series, field_name: str, rules: Dict) -> Dict[str, Any]:
        """
        Validate a single field according to its rules.
        
        Args:
            field_data: Series containing field data
            field_name: Name of the field
            rules: Validation rules for the field
            
        Returns:
            Dictionary containing field validation results
        """
        results = {
            'field_name': field_name,
            'total_values': len(field_data),
            'valid_values': 0,
            'null_values': 0,
            'invalid_values': 0,
            'error_details': []
        }
        
        # Count null values
        null_mask = field_data.isna()
        results['null_values'] = null_mask.sum()
        
        # Check if field is required and has nulls
        if rules.get('required', False) and results['null_values'] > 0:
            results['error_details'].append(f"Required field has {results['null_values']} null values")
        
        # Validate non-null values
        non_null_data = field_data[~null_mask]
        
        if len(non_null_data) > 0:
            valid_mask = self._get_field_valid_mask(non_null_data, field_name, rules)
            results['valid_values'] = valid_mask.sum()
            results['invalid_values'] = len(non_null_data) - results['valid_values']
            
            # Add specific error details
            if results['invalid_values'] > 0:
                invalid_data = non_null_data[~valid_mask]
                results['error_details'].append(
                    f"{results['invalid_values']} values failed validation rules"
                )
        
        # If nulls are allowed, count them as valid
        if not rules.get('required', False):
            results['valid_values'] += results['null_values']
        
        return results
    
    def _get_field_valid_mask(self, field_data: pd.Series, field_name: str, rules: Dict) -> pd.Series:
        """
        Get a boolean mask indicating which values in a field are valid.
        
        Args:
            field_data: Series containing field data (non-null values only)
            field_name: Name of the field
            rules: Validation rules for the field
            
        Returns:
            Boolean Series indicating valid values
        """
        valid_mask = pd.Series([True] * len(field_data), index=field_data.index)
        
        # Type validation
        if 'type' in rules:
            expected_types = rules['type'] if isinstance(rules['type'], tuple) else (rules['type'],)
            type_mask = field_data.apply(lambda x: isinstance(x, expected_types))
            valid_mask &= type_mask
        
        # String-specific validations
        if rules.get('type') == str or str in rules.get('type', []):
            # Length validations
            if 'min_length' in rules:
                length_mask = field_data.str.len() >= rules['min_length']
                valid_mask &= length_mask
            
            if 'max_length' in rules:
                length_mask = field_data.str.len() <= rules['max_length']
                valid_mask &= length_mask
            
            # Pattern validation
            if 'pattern' in rules:
                pattern_mask = field_data.str.match(rules['pattern'], na=False)
                valid_mask &= pattern_mask
        
        # Numeric-specific validations
        if any(t in [int, float] for t in (rules.get('type', []) if isinstance(rules.get('type'), tuple) else [rules.get('type')])):
            # Range validations
            if 'min_value' in rules:
                min_mask = field_data >= rules['min_value']
                valid_mask &= min_mask
            
            if 'max_value' in rules:
                max_mask = field_data <= rules['max_value']
                valid_mask &= max_mask
        
        return valid_mask
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Args:
            validation_results: Results from validate_data_quality
            
        Returns:
            Dictionary containing detailed validation report
        """
        report = {
            'summary': {
                'total_records_validated': validation_results['total_records'],
                'valid_records': validation_results['valid_records'],
                'invalid_records': validation_results['total_records'] - validation_results['valid_records'],
                'overall_quality_score': validation_results['overall_quality_score'],
                'duplicate_records': len(self.duplicate_order_ids)
            },
            'field_analysis': {},
            'data_quality_issues': [],
            'recommendations': []
        }
        
        # Analyze each field
        for field_name, field_results in validation_results['field_validation_results'].items():
            field_analysis = {
                'total_values': field_results['total_values'],
                'valid_values': field_results['valid_values'],
                'null_values': field_results['null_values'],
                'invalid_values': field_results['invalid_values'],
                'validity_rate': field_results['valid_values'] / field_results['total_values'] if field_results['total_values'] > 0 else 0,
                'null_rate': field_results['null_values'] / field_results['total_values'] if field_results['total_values'] > 0 else 0,
                'error_details': field_results['error_details']
            }
            report['field_analysis'][field_name] = field_analysis
            
            # Identify data quality issues
            if field_analysis['validity_rate'] < 0.95:  # Less than 95% valid
                report['data_quality_issues'].append(
                    f"Field '{field_name}' has low validity rate: {field_analysis['validity_rate']:.1%}"
                )
            
            if field_analysis['null_rate'] > 0.1:  # More than 10% null
                report['data_quality_issues'].append(
                    f"Field '{field_name}' has high null rate: {field_analysis['null_rate']:.1%}"
                )
        
        # Generate recommendations
        report['recommendations'] = self._generate_validation_recommendations(report)
        
        return report
    
    def _generate_validation_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Overall quality recommendations
        if report['summary']['overall_quality_score'] < 0.9:
            recommendations.append(
                "Overall data quality is below 90%. Consider implementing stricter data entry validation."
            )
        
        # Field-specific recommendations
        for field_name, analysis in report['field_analysis'].items():
            if analysis['validity_rate'] < 0.8:
                recommendations.append(
                    f"Field '{field_name}' has very low validity rate ({analysis['validity_rate']:.1%}). "
                    f"Review data collection process for this field."
                )
            
            if analysis['null_rate'] > 0.2:
                recommendations.append(
                    f"Field '{field_name}' has high null rate ({analysis['null_rate']:.1%}). "
                    f"Consider making this field required or providing default values."
                )
        
        # Duplicate recommendations
        if report['summary']['duplicate_records'] > 0:
            recommendations.append(
                f"Found {report['summary']['duplicate_records']} duplicate order IDs. "
                f"Implement unique constraint validation at data entry point."
            )
        
        if not recommendations:
            recommendations.append("Data quality appears excellent. Continue current data management practices.")
        
        return recommendations
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive validation statistics.
        
        Returns:
            Dictionary containing validation statistics
        """
        return {
            'total_records_processed': self.validation_stats['total_records'],
            'valid_records': self.validation_stats['valid_records'],
            'duplicate_records': self.validation_stats['duplicate_records'],
            'revenue_calculations_performed': self.validation_stats['revenue_calculations'],
            'unique_duplicate_order_ids': len(self.duplicate_order_ids),
            'validation_success_rate': (
                self.validation_stats['valid_records'] / self.validation_stats['total_records']
                if self.validation_stats['total_records'] > 0 else 0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            'total_records': 0,
            'valid_records': 0,
            'duplicate_records': 0,
            'validation_errors': {},
            'revenue_calculations': 0
        }
        self.duplicate_order_ids = set()
    
    def validate_chunk(self, chunk: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate a chunk of data with all validation checks.
        
        Args:
            chunk: DataFrame chunk to validate
            
        Returns:
            Tuple of (validated DataFrame, validation results)
        """
        logger.info(f"Validating chunk with {len(chunk)} records")
        
        # Detect duplicates
        chunk_with_duplicates = self.detect_duplicate_order_ids(chunk)
        
        # Calculate revenue
        chunk_with_revenue = self.calculate_revenue(chunk_with_duplicates)
        
        # Perform quality validation
        validation_results = self.validate_data_quality(chunk_with_revenue)
        
        logger.info(f"Chunk validation completed")
        
        return chunk_with_revenue, validation_results