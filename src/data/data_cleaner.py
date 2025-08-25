"""
Main data cleaning orchestrator.

This module provides the DataCleaner class that orchestrates the entire
data cleaning process using various field-specific cleaners and validators.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from email_validator import validate_email, EmailNotValidError

from .field_standardizer import FieldStandardizer

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Main orchestrator for data cleaning operations.
    
    This class coordinates the cleaning of various fields in e-commerce data
    including quantity validation, discount validation, email validation,
    and unit price validation.
    """
    
    def __init__(self):
        """Initialize the DataCleaner with field standardizer."""
        self.field_standardizer = FieldStandardizer()
        self.cleaning_stats = {
            'total_records': 0,
            'cleaned_records': 0,
            'errors_by_field': {},
            'corrections_by_field': {}
        }
    
    def clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Clean a chunk of data applying all cleaning rules.
        
        Args:
            chunk: DataFrame chunk to clean
            
        Returns:
            Cleaned DataFrame chunk
        """
        logger.info(f"Cleaning chunk with {len(chunk)} records")
        
        # Make a copy to avoid modifying original data
        cleaned_chunk = chunk.copy()
        
        # Update total records count
        self.cleaning_stats['total_records'] += len(chunk)
        
        # Apply field-specific cleaning
        if 'product_name' in cleaned_chunk.columns:
            cleaned_chunk['product_name'] = self.field_standardizer.standardize_product_names(
                cleaned_chunk['product_name']
            )
        
        if 'category' in cleaned_chunk.columns:
            cleaned_chunk['category'] = self.field_standardizer.standardize_categories(
                cleaned_chunk['category']
            )
        
        if 'region' in cleaned_chunk.columns:
            cleaned_chunk['region'] = self.field_standardizer.standardize_regions(
                cleaned_chunk['region']
            )
        
        if 'sale_date' in cleaned_chunk.columns:
            cleaned_chunk['sale_date'] = self.field_standardizer.parse_dates(
                cleaned_chunk['sale_date']
            )
        
        # Apply validation and conversion rules
        if 'quantity' in cleaned_chunk.columns:
            cleaned_chunk['quantity'] = self.validate_and_convert_quantities(
                cleaned_chunk['quantity']
            )
        
        if 'discount_percent' in cleaned_chunk.columns:
            cleaned_chunk['discount_percent'] = self.validate_discount_percentages(
                cleaned_chunk['discount_percent']
            )
        
        if 'customer_email' in cleaned_chunk.columns:
            cleaned_chunk['customer_email'] = self.validate_emails(
                cleaned_chunk['customer_email']
            )
        
        if 'unit_price' in cleaned_chunk.columns:
            cleaned_chunk['unit_price'] = self.validate_unit_prices(
                cleaned_chunk['unit_price']
            )
        
        # Update cleaned records count
        self.cleaning_stats['cleaned_records'] += len(cleaned_chunk)
        
        logger.info(f"Completed cleaning chunk")
        return cleaned_chunk
    
    def validate_and_convert_quantities(self, quantities: pd.Series) -> pd.Series:
        """
        Validate and convert quantity values from strings to numbers.
        
        Args:
            quantities: Series containing quantity values to validate
            
        Returns:
            Series with validated and converted quantities
        """
        logger.info(f"Validating {len(quantities)} quantity values")
        
        def clean_quantity(qty: Union[str, int, float]) -> int:
            """Clean and validate individual quantity value."""
            if pd.isna(qty):
                return 1  # Default quantity
            
            # Convert to string first to handle various input types
            qty_str = str(qty).strip().lower()
            
            # Handle empty strings
            if qty_str == '' or qty_str == 'nan':
                return 1
            
            # Remove common non-numeric characters
            qty_str = re.sub(r'[^\d\.\-\+]', '', qty_str)
            
            try:
                # Convert to float first, then to int
                qty_float = float(qty_str)
                qty_int = int(qty_float)
                
                # Validate range - quantities should be positive
                if qty_int <= 0:
                    logger.warning(f"Invalid quantity {qty_int}, setting to 1")
                    self._update_error_stats('quantity', 'non_positive')
                    return 1
                
                # Check for unreasonably high quantities (potential data error)
                if qty_int > 10000:
                    logger.warning(f"Unusually high quantity {qty_int}, capping at 10000")
                    self._update_correction_stats('quantity', 'capped_high')
                    return 10000
                
                return qty_int
                
            except (ValueError, TypeError):
                logger.warning(f"Could not convert quantity '{qty}' to number, setting to 1")
                self._update_error_stats('quantity', 'conversion_error')
                return 1
        
        cleaned_quantities = quantities.apply(clean_quantity)
        logger.info(f"Completed quantity validation")
        return cleaned_quantities
    
    def validate_discount_percentages(self, discounts: pd.Series) -> pd.Series:
        """
        Validate discount percentage values (should be between 0.0 and 1.0).
        
        Args:
            discounts: Series containing discount percentage values
            
        Returns:
            Series with validated discount percentages
        """
        logger.info(f"Validating {len(discounts)} discount percentage values")
        
        def clean_discount(discount: Union[str, int, float]) -> float:
            """Clean and validate individual discount percentage."""
            if pd.isna(discount):
                return 0.0  # Default no discount
            
            # Convert to string first to handle various input types
            discount_str = str(discount).strip().lower()
            
            # Handle empty strings
            if discount_str == '' or discount_str == 'nan':
                return 0.0
            
            # Remove percentage signs and other non-numeric characters except decimal point
            discount_str = re.sub(r'[^\d\.\-\+]', '', discount_str)
            
            try:
                discount_float = float(discount_str)
                
                # Handle percentages given as whole numbers (e.g., 15 instead of 0.15)
                if discount_float > 1.0 and discount_float <= 100.0:
                    discount_float = discount_float / 100.0
                    self._update_correction_stats('discount_percent', 'converted_from_percentage')
                
                # Validate range - discounts should be between 0.0 and 1.0
                if discount_float < 0.0:
                    logger.warning(f"Negative discount {discount_float}, setting to 0.0")
                    self._update_error_stats('discount_percent', 'negative_value')
                    return 0.0
                
                if discount_float > 1.0:
                    logger.warning(f"Discount > 100% ({discount_float}), capping at 1.0")
                    self._update_error_stats('discount_percent', 'over_100_percent')
                    return 1.0
                
                return discount_float
                
            except (ValueError, TypeError):
                logger.warning(f"Could not convert discount '{discount}' to number, setting to 0.0")
                self._update_error_stats('discount_percent', 'conversion_error')
                return 0.0
        
        cleaned_discounts = discounts.apply(clean_discount)
        logger.info(f"Completed discount percentage validation")
        return cleaned_discounts
    
    def validate_emails(self, emails: pd.Series) -> pd.Series:
        """
        Validate email addresses and handle null values.
        
        Args:
            emails: Series containing email addresses to validate
            
        Returns:
            Series with validated email addresses
        """
        logger.info(f"Validating {len(emails)} email addresses")
        
        def clean_email(email: Union[str, float]) -> Optional[str]:
            """Clean and validate individual email address."""
            if pd.isna(email) or email == '':
                return None  # Allow null emails
            
            # Convert to string and strip whitespace
            email_str = str(email).strip().lower()
            
            # Basic format check first
            if '@' not in email_str or '.' not in email_str:
                logger.warning(f"Invalid email format: {email_str}")
                self._update_error_stats('customer_email', 'invalid_format')
                return None
            
            try:
                # Use email-validator library for comprehensive validation
                validated_email = validate_email(email_str, check_deliverability=False)
                return validated_email.normalized
                
            except EmailNotValidError:
                logger.warning(f"Invalid email address: {email_str}")
                self._update_error_stats('customer_email', 'validation_failed')
                return None
        
        cleaned_emails = emails.apply(clean_email)
        logger.info(f"Completed email validation")
        return cleaned_emails
    
    def validate_unit_prices(self, prices: pd.Series) -> pd.Series:
        """
        Validate unit price values for negative or zero prices.
        
        Args:
            prices: Series containing unit price values
            
        Returns:
            Series with validated unit prices
        """
        logger.info(f"Validating {len(prices)} unit price values")
        
        def clean_unit_price(price: Union[str, int, float]) -> float:
            """Clean and validate individual unit price."""
            if pd.isna(price):
                logger.warning("Null unit price found, setting to 0.01")
                self._update_error_stats('unit_price', 'null_value')
                return 0.01  # Minimum price
            
            # Convert to string first to handle various input types
            price_str = str(price).strip()
            
            # Handle empty strings
            if price_str == '' or price_str.lower() == 'nan':
                logger.warning("Empty unit price found, setting to 0.01")
                self._update_error_stats('unit_price', 'empty_value')
                return 0.01
            
            # Remove currency symbols and other non-numeric characters except decimal point
            price_str = re.sub(r'[^\d\.\-\+]', '', price_str)
            
            try:
                price_float = float(price_str)
                
                # Validate range - prices should be positive
                if price_float <= 0.0:
                    logger.warning(f"Non-positive unit price {price_float}, setting to 0.01")
                    self._update_error_stats('unit_price', 'non_positive')
                    return 0.01
                
                # Check for unreasonably high prices (potential data error)
                if price_float > 1000000.0:
                    logger.warning(f"Unusually high unit price {price_float}, capping at 1000000.0")
                    self._update_correction_stats('unit_price', 'capped_high')
                    return 1000000.0
                
                return price_float
                
            except (ValueError, TypeError):
                logger.warning(f"Could not convert unit price '{price}' to number, setting to 0.01")
                self._update_error_stats('unit_price', 'conversion_error')
                return 0.01
        
        cleaned_prices = prices.apply(clean_unit_price)
        logger.info(f"Completed unit price validation")
        return cleaned_prices
    
    def _update_error_stats(self, field: str, error_type: str) -> None:
        """Update error statistics for tracking."""
        if field not in self.cleaning_stats['errors_by_field']:
            self.cleaning_stats['errors_by_field'][field] = {}
        
        if error_type not in self.cleaning_stats['errors_by_field'][field]:
            self.cleaning_stats['errors_by_field'][field][error_type] = 0
        
        self.cleaning_stats['errors_by_field'][field][error_type] += 1
    
    def _update_correction_stats(self, field: str, correction_type: str) -> None:
        """Update correction statistics for tracking."""
        if field not in self.cleaning_stats['corrections_by_field']:
            self.cleaning_stats['corrections_by_field'][field] = {}
        
        if correction_type not in self.cleaning_stats['corrections_by_field'][field]:
            self.cleaning_stats['corrections_by_field'][field][correction_type] = 0
        
        self.cleaning_stats['corrections_by_field'][field][correction_type] += 1
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cleaning statistics.
        
        Returns:
            Dictionary containing cleaning statistics
        """
        return {
            'total_records_processed': self.cleaning_stats['total_records'],
            'successfully_cleaned': self.cleaning_stats['cleaned_records'],
            'success_rate': (
                self.cleaning_stats['cleaned_records'] / self.cleaning_stats['total_records']
                if self.cleaning_stats['total_records'] > 0 else 0
            ),
            'errors_by_field': self.cleaning_stats['errors_by_field'],
            'corrections_by_field': self.cleaning_stats['corrections_by_field']
        }
    
    def reset_stats(self) -> None:
        """Reset cleaning statistics."""
        self.cleaning_stats = {
            'total_records': 0,
            'cleaned_records': 0,
            'errors_by_field': {},
            'corrections_by_field': {}
        }
    
    def apply_cleaning_rules(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning rules to a data chunk.
        
        This is an alias for clean_chunk for backward compatibility.
        
        Args:
            chunk: DataFrame chunk to clean
            
        Returns:
            Cleaned DataFrame chunk
        """
        return self.clean_chunk(chunk)
    
    def generate_cleaning_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive cleaning report.
        
        Returns:
            Dictionary containing detailed cleaning report
        """
        stats = self.get_cleaning_stats()
        
        # Calculate error rates by field
        error_rates = {}
        for field, errors in stats['errors_by_field'].items():
            total_errors = sum(errors.values())
            error_rates[field] = {
                'total_errors': total_errors,
                'error_rate': total_errors / stats['total_records_processed'] if stats['total_records_processed'] > 0 else 0,
                'error_breakdown': errors
            }
        
        # Calculate correction rates by field
        correction_rates = {}
        for field, corrections in stats['corrections_by_field'].items():
            total_corrections = sum(corrections.values())
            correction_rates[field] = {
                'total_corrections': total_corrections,
                'correction_rate': total_corrections / stats['total_records_processed'] if stats['total_records_processed'] > 0 else 0,
                'correction_breakdown': corrections
            }
        
        return {
            'summary': {
                'total_records_processed': stats['total_records_processed'],
                'successfully_cleaned': stats['successfully_cleaned'],
                'overall_success_rate': stats['success_rate']
            },
            'field_error_analysis': error_rates,
            'field_correction_analysis': correction_rates,
            'recommendations': self._generate_recommendations(error_rates, correction_rates)
        }
    
    def _generate_recommendations(self, error_rates: Dict, correction_rates: Dict) -> List[str]:
        """Generate recommendations based on cleaning statistics."""
        recommendations = []
        
        # Check for high error rates
        for field, stats in error_rates.items():
            if stats['error_rate'] > 0.1:  # More than 10% error rate
                recommendations.append(
                    f"High error rate ({stats['error_rate']:.1%}) in {field} field. "
                    f"Consider improving data collection or validation for this field."
                )
        
        # Check for common error patterns
        for field, stats in error_rates.items():
            if 'conversion_error' in stats['error_breakdown'] and stats['error_breakdown']['conversion_error'] > 100:
                recommendations.append(
                    f"Many conversion errors in {field} field. "
                    f"Review data format consistency and consider additional preprocessing."
                )
        
        # Check for high correction rates
        for field, stats in correction_rates.items():
            if stats['correction_rate'] > 0.2:  # More than 20% correction rate
                recommendations.append(
                    f"High correction rate ({stats['correction_rate']:.1%}) in {field} field. "
                    f"Consider updating data entry guidelines to reduce need for corrections."
                )
        
        if not recommendations:
            recommendations.append("Data quality appears good. Continue monitoring for any emerging patterns.")
        
        return recommendations