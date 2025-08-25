"""
Field standardization utilities for data cleaning.

This module provides field-specific cleaning and standardization logic
for e-commerce data processing.
"""

import re
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FieldStandardizer:
    """
    Handles field-specific cleaning and standardization operations.
    
    This class provides methods to clean and standardize various fields
    in e-commerce data including product names, categories, regions, and dates.
    """
    
    def __init__(self):
        """Initialize the FieldStandardizer with predefined mapping rules."""
        self._setup_category_mappings()
        self._setup_region_mappings()
        self._setup_date_formats()
    
    def _setup_category_mappings(self) -> None:
        """Set up category standardization mappings."""
        self.category_mappings = {
            # Electronics variations
            'electronics': 'Electronics',
            'electronic': 'Electronics',
            'elec': 'Electronics',
            'tech': 'Electronics',
            'technology': 'Electronics',
            
            # Fashion variations
            'fashion': 'Fashion',
            'clothing': 'Fashion',
            'apparel': 'Fashion',
            'clothes': 'Fashion',
            'wear': 'Fashion',
            
            # Home goods variations
            'home': 'Home Goods',
            'home goods': 'Home Goods',
            'homegoods': 'Home Goods',
            'home_goods': 'Home Goods',
            'household': 'Home Goods',
            'furniture': 'Home Goods',
            'decor': 'Home Goods',
            'home decor': 'Home Goods',
            
            # Beauty variations
            'beauty': 'Beauty',
            'cosmetics': 'Beauty',
            'skincare': 'Beauty',
            'makeup': 'Beauty',
            
            # Sports variations
            'sports': 'Sports',
            'sport': 'Sports',
            'fitness': 'Sports',
            'outdoor': 'Sports',
            
            # Books variations
            'books': 'Books',
            'book': 'Books',
            'literature': 'Books',
            'reading': 'Books'
        }
    
    def _setup_region_mappings(self) -> None:
        """Set up region standardization mappings."""
        self.region_mappings = {
            # North variations
            'north': 'North',
            'nort': 'North',
            'northern': 'North',
            'n': 'North',
            
            # South variations
            'south': 'South',
            'sout': 'South',
            'southern': 'South',
            's': 'South',
            
            # East variations
            'east': 'East',
            'eastern': 'East',
            'e': 'East',
            
            # West variations
            'west': 'West',
            'western': 'West',
            'w': 'West',
            
            # Central variations
            'central': 'Central',
            'centre': 'Central',
            'center': 'Central',
            'c': 'Central',
            
            # Southeast Asia specific regions
            'southeast': 'Southeast',
            'south east': 'Southeast',
            'south-east': 'Southeast',
            'se': 'Southeast',
            
            # Northeast variations
            'northeast': 'Northeast',
            'north east': 'Northeast',
            'north-east': 'Northeast',
            'ne': 'Northeast',
            
            # Northwest variations
            'northwest': 'Northwest',
            'north west': 'Northwest',
            'north-west': 'Northwest',
            'nw': 'Northwest',
            
            # Southwest variations
            'southwest': 'Southwest',
            'south west': 'Southwest',
            'south-west': 'Southwest',
            'sw': 'Southwest'
        }
    
    def _setup_date_formats(self) -> None:
        """Set up supported date formats for parsing."""
        self.date_formats = [
            '%Y-%m-%d',           # 2023-01-15
            '%d/%m/%Y',           # 15/01/2023
            '%m/%d/%Y',           # 01/15/2023
            '%d-%m-%Y',           # 15-01-2023
            '%m-%d-%Y',           # 01-15-2023
            '%Y/%m/%d',           # 2023/01/15
            '%d.%m.%Y',           # 15.01.2023
            '%Y-%m-%d %H:%M:%S',  # 2023-01-15 10:30:00
            '%d/%m/%Y %H:%M:%S',  # 15/01/2023 10:30:00
            '%m/%d/%Y %H:%M:%S',  # 01/15/2023 10:30:00
            '%Y-%m-%dT%H:%M:%S',  # 2023-01-15T10:30:00
            '%Y-%m-%dT%H:%M:%SZ', # 2023-01-15T10:30:00Z
        ]
    
    def standardize_product_names(self, names: pd.Series) -> pd.Series:
        """
        Clean and standardize product names.
        
        Args:
            names: Series containing product names to clean
            
        Returns:
            Series with cleaned product names
        """
        logger.info(f"Standardizing {len(names)} product names")
        
        def clean_product_name(name: Union[str, float]) -> str:
            """Clean individual product name."""
            if pd.isna(name) or name == '':
                return 'Unknown Product'
            
            # Convert to string and strip whitespace
            name = str(name).strip()
            
            # Remove extra whitespace and normalize spacing
            name = re.sub(r'\s+', ' ', name)
            
            # Remove special characters but keep alphanumeric, spaces, hyphens, underscores, parentheses, dots, commas, and ampersands
            name = re.sub(r'[^\w\s\-\(\)\.\,\&]', '', name)
            
            # Capitalize first letter of each word while preserving hyphens and underscores
            words = name.split()
            capitalized_words = []
            for word in words:
                # Handle words with hyphens and underscores
                if '-' in word or '_' in word:
                    # Split by hyphens and underscores, capitalize each part, then rejoin
                    parts = re.split(r'([-_])', word)
                    capitalized_parts = []
                    for part in parts:
                        if part in ['-', '_']:
                            capitalized_parts.append(part)
                        else:
                            capitalized_parts.append(part.capitalize())
                    capitalized_words.append(''.join(capitalized_parts))
                else:
                    capitalized_words.append(word.capitalize())
            name = ' '.join(capitalized_words)
            
            # Handle common abbreviations and expansions
            replacements = {
                ' And ': ' & ',
                ' With ': ' w/ ',
                ' For ': ' for ',
                ' The ': ' the ',
                ' Of ': ' of ',
                ' In ': ' in ',
                ' On ': ' on ',
                ' At ': ' at ',
                ' By ': ' by ',
                ' To ': ' to ',
            }
            
            for old, new in replacements.items():
                name = name.replace(old, new)
            
            return name.strip()
        
        cleaned_names = names.apply(clean_product_name)
        logger.info(f"Completed product name standardization")
        return cleaned_names
    
    def standardize_categories(self, categories: pd.Series) -> pd.Series:
        """
        Standardize category names using predefined mappings.
        
        Args:
            categories: Series containing category names to standardize
            
        Returns:
            Series with standardized category names
        """
        logger.info(f"Standardizing {len(categories)} categories")
        
        def clean_category(category: Union[str, float]) -> str:
            """Clean and standardize individual category."""
            if pd.isna(category) or category == '':
                return 'Other'
            
            # Convert to string, lowercase, and strip
            category = str(category).lower().strip()
            
            # Remove extra whitespace
            category = re.sub(r'\s+', ' ', category)
            
            # Remove special characters
            category = re.sub(r'[^\w\s]', '', category)
            
            # Check mappings
            if category in self.category_mappings:
                return self.category_mappings[category]
            
            # If no mapping found, capitalize first letter of each word and handle underscores
            words = category.split()
            capitalized_words = []
            for word in words:
                if '_' in word:
                    # Split by underscores, capitalize each part, then rejoin with spaces
                    parts = word.split('_')
                    capitalized_words.extend([part.capitalize() for part in parts if part])
                else:
                    capitalized_words.append(word.capitalize())
            return ' '.join(capitalized_words)
        
        standardized_categories = categories.apply(clean_category)
        logger.info(f"Completed category standardization")
        return standardized_categories
    
    def standardize_regions(self, regions: pd.Series) -> pd.Series:
        """
        Standardize region names using predefined mappings.
        
        Args:
            regions: Series containing region names to standardize
            
        Returns:
            Series with standardized region names
        """
        logger.info(f"Standardizing {len(regions)} regions")
        
        def clean_region(region: Union[str, float]) -> str:
            """Clean and standardize individual region."""
            if pd.isna(region) or region == '':
                return 'Unknown'
            
            # Convert to string, lowercase, and strip
            region = str(region).lower().strip()
            
            # Remove extra whitespace and special characters
            region = re.sub(r'\s+', ' ', region)
            region = re.sub(r'[^\w\s\-]', '', region)
            
            # Check mappings
            if region in self.region_mappings:
                return self.region_mappings[region]
            
            # If no mapping found, capitalize first letter
            return region.capitalize()
        
        standardized_regions = regions.apply(clean_region)
        logger.info(f"Completed region standardization")
        return standardized_regions
    
    def parse_dates(self, dates: pd.Series) -> pd.Series:
        """
        Parse dates from multiple formats and handle null values.
        
        Args:
            dates: Series containing date strings to parse
            
        Returns:
            Series with parsed datetime objects
        """
        logger.info(f"Parsing {len(dates)} dates")
        
        def parse_single_date(date_str: Union[str, float]) -> Optional[datetime]:
            """Parse individual date string."""
            if pd.isna(date_str) or date_str == '':
                return None
            
            # Convert to string and strip
            date_str = str(date_str).strip()
            
            # Try each format
            for fmt in self.date_formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # If no format works, try pandas to_datetime as fallback
            try:
                return pd.to_datetime(date_str)
            except:
                logger.warning(f"Could not parse date: {date_str}")
                return None
        
        parsed_dates = dates.apply(parse_single_date)
        
        # Count successful parses
        successful_parses = parsed_dates.notna().sum()
        total_dates = len(dates)
        logger.info(f"Successfully parsed {successful_parses}/{total_dates} dates")
        
        return parsed_dates
    
    def get_standardization_stats(self, original_data: pd.DataFrame, 
                                cleaned_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate statistics about the standardization process.
        
        Args:
            original_data: Original DataFrame before cleaning
            cleaned_data: DataFrame after cleaning
            
        Returns:
            Dictionary containing standardization statistics
        """
        stats = {}
        
        # Product name stats
        if 'product_name' in original_data.columns and 'product_name' in cleaned_data.columns:
            original_unique = original_data['product_name'].nunique()
            cleaned_unique = cleaned_data['product_name'].nunique()
            stats['product_names'] = {
                'original_unique_count': original_unique,
                'cleaned_unique_count': cleaned_unique,
                'reduction_ratio': (original_unique - cleaned_unique) / original_unique if original_unique > 0 else 0
            }
        
        # Category stats
        if 'category' in original_data.columns and 'category' in cleaned_data.columns:
            original_categories = set(original_data['category'].dropna().str.lower())
            cleaned_categories = set(cleaned_data['category'].dropna())
            stats['categories'] = {
                'original_unique_count': len(original_categories),
                'cleaned_unique_count': len(cleaned_categories),
                'standardized_categories': list(cleaned_categories)
            }
        
        # Region stats
        if 'region' in original_data.columns and 'region' in cleaned_data.columns:
            original_regions = set(original_data['region'].dropna().str.lower())
            cleaned_regions = set(cleaned_data['region'].dropna())
            stats['regions'] = {
                'original_unique_count': len(original_regions),
                'cleaned_unique_count': len(cleaned_regions),
                'standardized_regions': list(cleaned_regions)
            }
        
        # Date parsing stats
        if 'sale_date' in original_data.columns and 'sale_date' in cleaned_data.columns:
            original_nulls = original_data['sale_date'].isna().sum()
            cleaned_nulls = cleaned_data['sale_date'].isna().sum()
            stats['dates'] = {
                'original_null_count': int(original_nulls),
                'cleaned_null_count': int(cleaned_nulls),
                'successfully_parsed': int(len(original_data) - cleaned_nulls),
                'parse_success_rate': (len(original_data) - cleaned_nulls) / len(original_data) if len(original_data) > 0 else 0
            }
        
        return stats