"""
Unit tests for FieldStandardizer class.
"""

import pytest
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.field_standardizer import FieldStandardizer


class TestFieldStandardizer:
    """Test cases for FieldStandardizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.standardizer = FieldStandardizer()
    
    def test_standardize_product_names_basic(self):
        """Test basic product name standardization."""
        names = pd.Series([
            'apple iphone 13',
            'SAMSUNG GALAXY S21',
            'nike air max shoes',
            '  extra   spaces  ',
            'product@#$%with&*special()chars',
            '',
            None
        ])
        
        result = self.standardizer.standardize_product_names(names)
        
        expected = [
            'Apple Iphone 13',
            'Samsung Galaxy S21',
            'Nike Air Max Shoes',
            'Extra Spaces',
            'Productwith&special()chars',
            'Unknown Product',
            'Unknown Product'
        ]
        
        assert result.tolist() == expected
    
    def test_standardize_categories_mapping(self):
        """Test category standardization with mappings."""
        categories = pd.Series([
            'electronics',
            'FASHION',
            'home goods',
            'elec',
            'clothing',
            'tech',
            'unknown_category',
            '',
            None
        ])
        
        result = self.standardizer.standardize_categories(categories)
        
        expected = [
            'Electronics',
            'Fashion',
            'Home Goods',
            'Electronics',
            'Fashion',
            'Electronics',
            'Unknown Category',
            'Other',
            'Other'
        ]
        
        assert result.tolist() == expected
    
    def test_standardize_regions_mapping(self):
        """Test region standardization with mappings."""
        regions = pd.Series([
            'north',
            'SOUTH',
            'nort',
            'eastern',
            'w',
            'southeast',
            'unknown_region',
            '',
            None
        ])
        
        result = self.standardizer.standardize_regions(regions)
        
        expected = [
            'North',
            'South',
            'North',
            'East',
            'West',
            'Southeast',
            'Unknown_region',
            'Unknown',
            'Unknown'
        ]
        
        assert result.tolist() == expected
    
    def test_parse_dates_multiple_formats(self):
        """Test date parsing with multiple formats."""
        dates = pd.Series([
            '2023-01-15',
            '15/01/2023',
            '01/15/2023',
            '2023/01/15',
            '15.01.2023',
            '2023-01-15 10:30:00',
            '2023-01-15T10:30:00',
            'invalid_date',
            '',
            None
        ])
        
        result = self.standardizer.parse_dates(dates)
        
        # Check that valid dates are parsed correctly
        assert result.iloc[0] == datetime(2023, 1, 15)
        assert result.iloc[1] == datetime(2023, 1, 15)
        assert result.iloc[5] == datetime(2023, 1, 15, 10, 30, 0)
        
        # Check that invalid dates are None
        assert pd.isna(result.iloc[7])  # invalid_date
        assert pd.isna(result.iloc[8])  # empty string
        assert pd.isna(result.iloc[9])  # None
    
    def test_product_names_edge_cases(self):
        """Test product name standardization edge cases."""
        names = pd.Series([
            'product with and in the name',
            'PRODUCT WITH MULTIPLE   SPACES',
            '123 numeric product',
            'a',  # single character
            'Product-Name_With-Special_Chars'
        ])
        
        result = self.standardizer.standardize_product_names(names)
        
        assert 'Product w/ & in the Name' == result.iloc[0]
        assert 'Product w/ Multiple Spaces' == result.iloc[1]
        assert '123 Numeric Product' == result.iloc[2]
        assert 'A' == result.iloc[3]
        assert 'Product-Name_With-Special_Chars' == result.iloc[4]
    
    def test_categories_case_insensitive(self):
        """Test that category mapping is case insensitive."""
        categories = pd.Series([
            'ELECTRONICS',
            'electronics',
            'Electronics',
            'eLeCtrOnIcS'
        ])
        
        result = self.standardizer.standardize_categories(categories)
        
        # All should map to 'Electronics'
        assert all(cat == 'Electronics' for cat in result)
    
    def test_regions_case_insensitive(self):
        """Test that region mapping is case insensitive."""
        regions = pd.Series([
            'NORTH',
            'north',
            'North',
            'NoRtH'
        ])
        
        result = self.standardizer.standardize_regions(regions)
        
        # All should map to 'North'
        assert all(region == 'North' for region in result)
    
    def test_get_standardization_stats(self):
        """Test standardization statistics generation."""
        original_data = pd.DataFrame({
            'product_name': ['apple iphone', 'APPLE IPHONE', 'samsung galaxy'],
            'category': ['electronics', 'elec', 'fashion'],
            'region': ['north', 'nort', 'south'],
            'sale_date': ['2023-01-15', '15/01/2023', 'invalid']
        })
        
        cleaned_data = pd.DataFrame({
            'product_name': ['Apple Iphone', 'Apple Iphone', 'Samsung Galaxy'],
            'category': ['Electronics', 'Electronics', 'Fashion'],
            'region': ['North', 'North', 'South'],
            'sale_date': [datetime(2023, 1, 15), datetime(2023, 1, 15), None]
        })
        
        stats = self.standardizer.get_standardization_stats(original_data, cleaned_data)
        
        # Check product name stats
        assert stats['product_names']['original_unique_count'] == 3
        assert stats['product_names']['cleaned_unique_count'] == 2
        
        # Check category stats
        assert stats['categories']['original_unique_count'] == 3
        assert stats['categories']['cleaned_unique_count'] == 2
        
        # Check region stats
        assert stats['regions']['original_unique_count'] == 3
        assert stats['regions']['cleaned_unique_count'] == 2
        
        # Check date stats
        assert stats['dates']['successfully_parsed'] == 2
        assert stats['dates']['parse_success_rate'] == 2/3
    
    def test_standardization_performance(self):
        """Test standardization performance with large datasets."""
        import time
        
        # Create large dataset
        large_data = pd.Series([f'product name {i}' for i in range(10000)])
        
        start_time = time.time()
        result = self.standardizer.standardize_product_names(large_data)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 5 seconds)
        assert (end_time - start_time) < 5.0
        assert len(result) == 10000
        assert all(isinstance(name, str) for name in result)
    
    def test_date_parsing_edge_cases(self):
        """Test date parsing with various edge cases."""
        dates = pd.Series([
            '2023-02-29',  # Invalid leap year date
            '2024-02-29',  # Valid leap year date
            '2023-13-01',  # Invalid month
            '2023-01-32',  # Invalid day
            '23-01-01',    # Two-digit year
            '2023/1/1',    # Single digit month/day
            '1/1/23',      # Ambiguous format
            '2023-01-01 25:00:00',  # Invalid time
            'Jan 1, 2023', # Text month
            '1st January 2023'  # Ordinal day
        ])
        
        result = self.standardizer.parse_dates(dates)
        
        # Check that invalid dates are None
        assert pd.isna(result.iloc[0])  # Invalid leap year
        assert result.iloc[1] == datetime(2024, 2, 29)  # Valid leap year
        assert pd.isna(result.iloc[2])  # Invalid month
        assert pd.isna(result.iloc[3])  # Invalid day
    
    def test_category_mapping_extensibility(self):
        """Test that category mapping can be extended."""
        # Test with new categories not in the default mapping
        categories = pd.Series([
            'automotive',
            'books',
            'sports',
            'music',
            'unknown_new_category'
        ])
        
        result = self.standardizer.standardize_categories(categories)
        
        # Should handle unknown categories gracefully
        assert 'Automotive' in result.values or 'Unknown Category' in result.values
        assert 'Books' in result.values or 'Unknown Category' in result.values
    
    def test_product_name_cleaning_rules(self):
        """Test specific product name cleaning rules."""
        names = pd.Series([
            'apple iphone 13 pro max',
            'SAMSUNG GALAXY S21 ULTRA',
            'nike air jordan 1 retro high',
            'sony wh-1000xm4 headphones',
            'dell xps 13 laptop computer'
        ])
        
        result = self.standardizer.standardize_product_names(names)
        
        # Check title case conversion
        assert result.iloc[0] == 'Apple Iphone 13 Pro Max'
        assert result.iloc[1] == 'Samsung Galaxy S21 Ultra'
        
        # Check that hyphens are preserved
        assert 'wh-1000xm4' in result.iloc[3].lower()
    
    def test_region_standardization_fuzzy_matching(self):
        """Test fuzzy matching for region standardization."""
        regions = pd.Series([
            'northen',  # Typo
            'sout',     # Partial
            'est',      # Partial
            'wst',      # Partial
            'northeast',
            'southwest',
            'central'
        ])
        
        result = self.standardizer.standardize_regions(regions)
        
        # Should handle typos and partial matches
        assert result.iloc[0] in ['North', 'Northeast', 'Unknown']
        assert result.iloc[1] in ['South', 'Southwest', 'Unknown']


if __name__ == '__main__':
    pytest.main([__file__])