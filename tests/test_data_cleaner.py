"""
Unit tests for DataCleaner class.

This module contains comprehensive tests for data cleaning functionality
including field standardization, validation, and edge case handling.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_cleaner import DataCleaner


class TestDataCleaner:
    """Test cases for DataCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = DataCleaner()
    
    def test_validate_and_convert_quantities_basic(self):
        """Test basic quantity validation and conversion."""
        quantities = pd.Series([
            '5',
            '10.5',
            '0',
            '-2',
            'abc',
            '',
            None,
            15,
            '20 units'
        ])
        
        result = self.cleaner.validate_and_convert_quantities(quantities)
        
        expected = [5, 10, 1, 1, 1, 1, 1, 15, 20]
        assert result.tolist() == expected
    
    def test_validate_and_convert_quantities_edge_cases(self):
        """Test quantity validation edge cases."""
        quantities = pd.Series([
            '15000',  # Very high quantity
            '0.9',    # Decimal that rounds down
            '1.9',    # Decimal that rounds down
            'nan'     # String 'nan'
        ])
        
        result = self.cleaner.validate_and_convert_quantities(quantities)
        
        expected = [10000, 1, 1, 1]  # High capped, decimals handled, nan handled
        assert result.tolist() == expected
    
    def test_validate_discount_percentages_basic(self):
        """Test basic discount percentage validation."""
        discounts = pd.Series([
            '0.15',
            '15',      # Should convert from percentage
            '0.5',
            '1.0',
            '150',     # Over 100%, should cap
            '-0.1',    # Negative, should set to 0
            'abc',
            '',
            None
        ])
        
        result = self.cleaner.validate_discount_percentages(discounts)
        
        expected = [0.15, 0.15, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        assert result.tolist() == expected
    
    def test_validate_discount_percentages_edge_cases(self):
        """Test discount percentage validation edge cases."""
        discounts = pd.Series([
            '25%',     # With percentage sign
            '0.999',   # Close to 1
            '100.1',   # Just over 100
            'nan'      # String 'nan'
        ])
        
        result = self.cleaner.validate_discount_percentages(discounts)
        
        expected = [0.25, 0.999, 1.0, 0.0]
        assert result.tolist() == expected
    
    def test_validate_emails_basic(self):
        """Test basic email validation."""
        emails = pd.Series([
            'user@example.com',
            'INVALID.EMAIL',
            'test@domain.co.uk',
            'no-at-sign.com',
            '',
            None,
            'user@',
            '@domain.com'
        ])
        
        result = self.cleaner.validate_emails(emails)
        
        expected = [
            'user@example.com',
            None,  # Invalid format
            'test@domain.co.uk',
            None,  # No @ sign
            None,  # Empty
            None,  # None
            None,  # Missing domain
            None   # Missing user
        ]
        assert result.tolist() == expected
    
    def test_validate_unit_prices_basic(self):
        """Test basic unit price validation."""
        prices = pd.Series([
            '10.99',
            '0',       # Zero price
            '-5.50',   # Negative price
            'abc',
            '',
            None,
            25.75,
            '$15.99',  # With currency symbol
            '2000000'  # Very high price
        ])
        
        result = self.cleaner.validate_unit_prices(prices)
        
        expected = [10.99, 0.01, 0.01, 0.01, 0.01, 0.01, 25.75, 15.99, 1000000.0]
        assert result.tolist() == expected
    
    def test_clean_chunk_integration(self):
        """Test complete chunk cleaning integration."""
        chunk = pd.DataFrame({
            'product_name': ['apple iphone', 'SAMSUNG GALAXY'],
            'category': ['electronics', 'tech'],
            'region': ['north', 'south'],
            'quantity': ['5', '0'],
            'discount_percent': ['15', '0.25'],
            'customer_email': ['user@test.com', 'invalid.email'],
            'unit_price': ['99.99', '-10'],
            'sale_date': ['2023-01-15', '15/01/2023']
        })
        
        result = self.cleaner.clean_chunk(chunk)
        
        # Check that all fields were processed
        assert result['product_name'].iloc[0] == 'Apple Iphone'
        assert result['category'].iloc[0] == 'Electronics'
        assert result['region'].iloc[0] == 'North'
        assert result['quantity'].iloc[0] == 5
        assert result['quantity'].iloc[1] == 1  # Zero converted to 1
        assert result['discount_percent'].iloc[0] == 0.15  # Converted from percentage
        assert result['customer_email'].iloc[0] == 'user@test.com'
        assert pd.isna(result['customer_email'].iloc[1])  # Invalid email set to None
        assert result['unit_price'].iloc[0] == 99.99
        assert result['unit_price'].iloc[1] == 0.01  # Negative price corrected
        assert isinstance(result['sale_date'].iloc[0], datetime)
    
    def test_cleaning_stats_tracking(self):
        """Test that cleaning statistics are properly tracked."""
        chunk = pd.DataFrame({
            'quantity': ['abc', '0', '-5'],
            'discount_percent': ['150', '-0.1', '0.25'],
            'unit_price': ['0', 'invalid', '10.99']
        })
        
        self.cleaner.clean_chunk(chunk)
        stats = self.cleaner.get_cleaning_stats()
        
        # Check that errors were tracked
        assert 'quantity' in stats['errors_by_field']
        assert 'discount_percent' in stats['errors_by_field']
        assert 'unit_price' in stats['errors_by_field']
        
        # Check total records
        assert stats['total_records_processed'] == 3
        assert stats['successfully_cleaned'] == 3
    
    def test_generate_cleaning_report(self):
        """Test cleaning report generation."""
        chunk = pd.DataFrame({
            'quantity': ['abc', '0', '5'],
            'discount_percent': ['150', '0.25', '15'],
            'unit_price': ['0', '10.99', 'invalid']
        })
        
        self.cleaner.clean_chunk(chunk)
        report = self.cleaner.generate_cleaning_report()
        
        # Check report structure
        assert 'summary' in report
        assert 'field_error_analysis' in report
        assert 'field_correction_analysis' in report
        assert 'recommendations' in report
        
        # Check summary
        assert report['summary']['total_records_processed'] == 3
        assert report['summary']['successfully_cleaned'] == 3
    
    def test_reset_stats(self):
        """Test statistics reset functionality."""
        chunk = pd.DataFrame({
            'quantity': ['abc', '5'],
            'unit_price': ['0', '10.99']
        })
        
        self.cleaner.clean_chunk(chunk)
        stats_before = self.cleaner.get_cleaning_stats()
        
        self.cleaner.reset_stats()
        stats_after = self.cleaner.get_cleaning_stats()
        
        # Check that stats were reset
        assert stats_before['total_records_processed'] == 2
        assert stats_after['total_records_processed'] == 0
        assert len(stats_after['errors_by_field']) == 0
    
    def test_apply_cleaning_rules_alias(self):
        """Test that apply_cleaning_rules works as alias for clean_chunk."""
        chunk = pd.DataFrame({
            'product_name': ['test product'],
            'quantity': ['5']
        })
        
        result1 = self.cleaner.clean_chunk(chunk.copy())
        result2 = self.cleaner.apply_cleaning_rules(chunk.copy())
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_extreme_edge_cases(self):
        """Test extreme edge cases for data cleaning."""
        # Test with extremely large numbers
        chunk = pd.DataFrame({
            'quantity': ['999999999999', '1e10', '1.5e15'],
            'unit_price': ['1e20', '999999999999.99', '0.000001'],
            'discount_percent': ['99.999', '0.001', '100.0']
        })
        
        result = self.cleaner.clean_chunk(chunk)
        
        # Check that extreme values are handled
        assert all(result['quantity'] <= 10000)  # Should be capped
        assert all(result['unit_price'] <= 1000000)  # Should be capped
        assert all(result['discount_percent'] <= 1.0)  # Should be capped
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        chunk = pd.DataFrame({
            'product_name': ['Café Münü', 'Naïve résumé', '测试产品', '🎉 Special Product'],
            'customer_email': ['café@münü.com', 'naïve@résumé.org', 'test@测试.com', 'emoji@🎉.com']
        })
        
        result = self.cleaner.clean_chunk(chunk)
        
        # Check that unicode characters are preserved in product names
        assert 'Café Münü' in result['product_name'].values
        assert 'Naïve Résumé' in result['product_name'].values
        
        # Check email validation handles unicode domains appropriately
        assert pd.isna(result['customer_email'].iloc[3])  # Emoji domain should be invalid
    
    def test_memory_efficient_processing(self):
        """Test memory-efficient processing of large chunks."""
        # Create a large chunk to test memory efficiency
        large_chunk = pd.DataFrame({
            'product_name': [f'Product {i}' for i in range(10000)],
            'quantity': [str(i % 100) for i in range(10000)],
            'unit_price': [str(10.0 + i % 1000) for i in range(10000)],
            'discount_percent': [str((i % 30) / 100) for i in range(10000)]
        })
        
        # This should not raise memory errors
        result = self.cleaner.clean_chunk(large_chunk)
        
        assert len(result) == 10000
        assert all(result['quantity'] > 0)
        assert all(result['unit_price'] > 0)
    
    def test_concurrent_cleaning_safety(self):
        """Test that cleaning operations are safe for concurrent use."""
        import threading
        import time
        
        results = []
        errors = []
        
        def clean_data(thread_id):
            try:
                chunk = pd.DataFrame({
                    'product_name': [f'Product {thread_id}_{i}' for i in range(100)],
                    'quantity': [str(i + 1) for i in range(100)]
                })
                result = self.cleaner.clean_chunk(chunk)
                results.append((thread_id, len(result)))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=clean_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(count == 100 for _, count in results)


if __name__ == '__main__':
    pytest.main([__file__])