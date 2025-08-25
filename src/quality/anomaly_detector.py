"""
Anomaly detection system for identifying suspicious transactions.

This module provides the AnomalyDetector class that identifies anomalous
transactions using statistical methods including z-scores and IQR analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Main anomaly detection engine for identifying suspicious transactions.
    
    This class implements statistical anomaly detection using z-scores and IQR
    methods to identify transactions with unusual patterns in revenue, quantity,
    and discount values.
    """
    
    def __init__(self, z_score_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        """
        Initialize the AnomalyDetector.
        
        Args:
            z_score_threshold: Z-score threshold for anomaly detection (default: 3.0)
            iqr_multiplier: IQR multiplier for outlier detection (default: 1.5)
        """
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.detection_stats = {
            'total_records_analyzed': 0,
            'revenue_anomalies': 0,
            'quantity_anomalies': 0,
            'discount_anomalies': 0,
            'total_anomalies': 0,
            'detection_errors': []
        }
        self.anomaly_records = []
    
    def detect_revenue_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create revenue anomaly detection for extremely high values.
        
        Args:
            data: DataFrame containing sales data with revenue column
            
        Returns:
            DataFrame with revenue anomaly flags and scores
            
        Requirements: 7.1
        """
        logger.info(f"Detecting revenue anomalies in {len(data)} records")
        
        # Update total records analyzed
        self.detection_stats['total_records_analyzed'] += len(data)
        
        try:
            # Ensure required columns exist
            if 'revenue' not in data.columns:
                # Calculate revenue if not present
                required_cols = ['quantity', 'unit_price', 'discount_percent']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns for revenue calculation: {missing_cols}")
                
                data = data.copy()
                data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])
            
            result_data = data.copy()
            
            # Remove null or invalid revenue values
            valid_revenue_mask = pd.notna(result_data['revenue']) & np.isfinite(result_data['revenue'])
            valid_data = result_data[valid_revenue_mask]
            
            if len(valid_data) == 0:
                logger.warning("No valid revenue data found for anomaly detection")
                result_data['revenue_anomaly_zscore'] = False
                result_data['revenue_anomaly_iqr'] = False
                result_data['revenue_anomaly_score'] = 0.0
                result_data['revenue_anomaly_reason'] = 'No valid data'
                return result_data
            
            # Z-score based anomaly detection
            revenue_mean = valid_data['revenue'].mean()
            revenue_std = valid_data['revenue'].std()
            
            if revenue_std > 0:
                z_scores = np.abs((valid_data['revenue'] - revenue_mean) / revenue_std)
                zscore_anomaly_mask = z_scores > self.z_score_threshold
            else:
                z_scores = pd.Series([0.0] * len(valid_data), index=valid_data.index)
                zscore_anomaly_mask = pd.Series([False] * len(valid_data), index=valid_data.index)
            
            # IQR based anomaly detection
            q1 = valid_data['revenue'].quantile(0.25)
            q3 = valid_data['revenue'].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - (self.iqr_multiplier * iqr)
            upper_bound = q3 + (self.iqr_multiplier * iqr)
            
            iqr_anomaly_mask = (valid_data['revenue'] < lower_bound) | (valid_data['revenue'] > upper_bound)
            
            # Initialize anomaly columns for all records
            result_data['revenue_anomaly_zscore'] = False
            result_data['revenue_anomaly_iqr'] = False
            result_data['revenue_anomaly_score'] = 0.0
            result_data['revenue_anomaly_reason'] = 'Normal'
            
            # Set anomaly flags for valid data
            result_data.loc[valid_data.index, 'revenue_anomaly_zscore'] = zscore_anomaly_mask
            result_data.loc[valid_data.index, 'revenue_anomaly_iqr'] = iqr_anomaly_mask
            result_data.loc[valid_data.index, 'revenue_anomaly_score'] = z_scores
            
            # Set anomaly reasons
            zscore_anomalies = result_data[result_data['revenue_anomaly_zscore']]
            iqr_anomalies = result_data[result_data['revenue_anomaly_iqr']]
            
            result_data.loc[zscore_anomalies.index, 'revenue_anomaly_reason'] = 'High Z-score'
            result_data.loc[iqr_anomalies.index, 'revenue_anomaly_reason'] = 'IQR Outlier'
            
            # Both methods detected anomaly
            both_anomalies = result_data[
                result_data['revenue_anomaly_zscore'] & result_data['revenue_anomaly_iqr']
            ]
            result_data.loc[both_anomalies.index, 'revenue_anomaly_reason'] = 'Z-score + IQR Outlier'
            
            # Update statistics
            revenue_anomaly_count = (
                result_data['revenue_anomaly_zscore'] | result_data['revenue_anomaly_iqr']
            ).sum()
            self.detection_stats['revenue_anomalies'] += revenue_anomaly_count
            
            logger.info(f"Detected {revenue_anomaly_count} revenue anomalies")
            
            return result_data
            
        except Exception as e:
            error_msg = f"Error detecting revenue anomalies: {str(e)}"
            logger.error(error_msg)
            self.detection_stats['detection_errors'].append(error_msg)
            raise
    
    def detect_quantity_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build quantity anomaly detection logic.
        
        Args:
            data: DataFrame containing sales data with quantity column
            
        Returns:
            DataFrame with quantity anomaly flags and scores
            
        Requirements: 7.2
        """
        logger.info(f"Detecting quantity anomalies in {len(data)} records")
        
        try:
            # Ensure required column exists
            if 'quantity' not in data.columns:
                raise ValueError("Missing required column: quantity")
            
            result_data = data.copy()
            
            # Remove null or invalid quantity values
            valid_quantity_mask = (
                pd.notna(result_data['quantity']) & 
                np.isfinite(result_data['quantity']) & 
                (result_data['quantity'] > 0)
            )
            valid_data = result_data[valid_quantity_mask]
            
            if len(valid_data) == 0:
                logger.warning("No valid quantity data found for anomaly detection")
                result_data['quantity_anomaly_zscore'] = False
                result_data['quantity_anomaly_iqr'] = False
                result_data['quantity_anomaly_score'] = 0.0
                result_data['quantity_anomaly_reason'] = 'No valid data'
                return result_data
            
            # Z-score based anomaly detection
            quantity_mean = valid_data['quantity'].mean()
            quantity_std = valid_data['quantity'].std()
            
            if quantity_std > 0:
                z_scores = np.abs((valid_data['quantity'] - quantity_mean) / quantity_std)
                zscore_anomaly_mask = z_scores > self.z_score_threshold
            else:
                z_scores = pd.Series([0.0] * len(valid_data), index=valid_data.index)
                zscore_anomaly_mask = pd.Series([False] * len(valid_data), index=valid_data.index)
            
            # IQR based anomaly detection
            q1 = valid_data['quantity'].quantile(0.25)
            q3 = valid_data['quantity'].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - (self.iqr_multiplier * iqr)
            upper_bound = q3 + (self.iqr_multiplier * iqr)
            
            # For quantities, we're mainly interested in unusually high values
            iqr_anomaly_mask = valid_data['quantity'] > upper_bound
            
            # Initialize anomaly columns for all records
            result_data['quantity_anomaly_zscore'] = False
            result_data['quantity_anomaly_iqr'] = False
            result_data['quantity_anomaly_score'] = 0.0
            result_data['quantity_anomaly_reason'] = 'Normal'
            
            # Set anomaly flags for valid data
            result_data.loc[valid_data.index, 'quantity_anomaly_zscore'] = zscore_anomaly_mask
            result_data.loc[valid_data.index, 'quantity_anomaly_iqr'] = iqr_anomaly_mask
            result_data.loc[valid_data.index, 'quantity_anomaly_score'] = z_scores
            
            # Set anomaly reasons
            zscore_anomalies = result_data[result_data['quantity_anomaly_zscore']]
            iqr_anomalies = result_data[result_data['quantity_anomaly_iqr']]
            
            result_data.loc[zscore_anomalies.index, 'quantity_anomaly_reason'] = 'High Z-score'
            result_data.loc[iqr_anomalies.index, 'quantity_anomaly_reason'] = 'IQR Outlier'
            
            # Both methods detected anomaly
            both_anomalies = result_data[
                result_data['quantity_anomaly_zscore'] & result_data['quantity_anomaly_iqr']
            ]
            result_data.loc[both_anomalies.index, 'quantity_anomaly_reason'] = 'Z-score + IQR Outlier'
            
            # Update statistics
            quantity_anomaly_count = (
                result_data['quantity_anomaly_zscore'] | result_data['quantity_anomaly_iqr']
            ).sum()
            self.detection_stats['quantity_anomalies'] += quantity_anomaly_count
            
            logger.info(f"Detected {quantity_anomaly_count} quantity anomalies")
            
            return result_data
            
        except Exception as e:
            error_msg = f"Error detecting quantity anomalies: {str(e)}"
            logger.error(error_msg)
            self.detection_stats['detection_errors'].append(error_msg)
            raise
    
    def detect_discount_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build discount anomaly detection logic.
        
        Args:
            data: DataFrame containing sales data with discount_percent column
            
        Returns:
            DataFrame with discount anomaly flags and scores
            
        Requirements: 7.2
        """
        logger.info(f"Detecting discount anomalies in {len(data)} records")
        
        try:
            # Ensure required column exists
            if 'discount_percent' not in data.columns:
                raise ValueError("Missing required column: discount_percent")
            
            result_data = data.copy()
            
            # Remove null or invalid discount values
            valid_discount_mask = (
                pd.notna(result_data['discount_percent']) & 
                np.isfinite(result_data['discount_percent']) & 
                (result_data['discount_percent'] >= 0) &
                (result_data['discount_percent'] <= 1)
            )
            valid_data = result_data[valid_discount_mask]
            
            if len(valid_data) == 0:
                logger.warning("No valid discount data found for anomaly detection")
                result_data['discount_anomaly_zscore'] = False
                result_data['discount_anomaly_iqr'] = False
                result_data['discount_anomaly_score'] = 0.0
                result_data['discount_anomaly_reason'] = 'No valid data'
                return result_data
            
            # Z-score based anomaly detection
            discount_mean = valid_data['discount_percent'].mean()
            discount_std = valid_data['discount_percent'].std()
            
            if discount_std > 0:
                z_scores = np.abs((valid_data['discount_percent'] - discount_mean) / discount_std)
                zscore_anomaly_mask = z_scores > self.z_score_threshold
            else:
                z_scores = pd.Series([0.0] * len(valid_data), index=valid_data.index)
                zscore_anomaly_mask = pd.Series([False] * len(valid_data), index=valid_data.index)
            
            # IQR based anomaly detection
            q1 = valid_data['discount_percent'].quantile(0.25)
            q3 = valid_data['discount_percent'].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - (self.iqr_multiplier * iqr)
            upper_bound = q3 + (self.iqr_multiplier * iqr)
            
            # For discounts, we're interested in both unusually high and low values
            iqr_anomaly_mask = (
                (valid_data['discount_percent'] < lower_bound) | 
                (valid_data['discount_percent'] > upper_bound)
            )
            
            # Initialize anomaly columns for all records
            result_data['discount_anomaly_zscore'] = False
            result_data['discount_anomaly_iqr'] = False
            result_data['discount_anomaly_score'] = 0.0
            result_data['discount_anomaly_reason'] = 'Normal'
            
            # Set anomaly flags for valid data
            result_data.loc[valid_data.index, 'discount_anomaly_zscore'] = zscore_anomaly_mask
            result_data.loc[valid_data.index, 'discount_anomaly_iqr'] = iqr_anomaly_mask
            result_data.loc[valid_data.index, 'discount_anomaly_score'] = z_scores
            
            # Set anomaly reasons with more specific descriptions
            high_discount_mask = valid_data['discount_percent'] > upper_bound
            low_discount_mask = valid_data['discount_percent'] < lower_bound
            
            result_data.loc[
                valid_data[zscore_anomaly_mask].index, 'discount_anomaly_reason'
            ] = 'High Z-score'
            
            result_data.loc[
                valid_data[high_discount_mask].index, 'discount_anomaly_reason'
            ] = 'Unusually High Discount'
            
            result_data.loc[
                valid_data[low_discount_mask].index, 'discount_anomaly_reason'
            ] = 'Unusually Low Discount'
            
            # Both methods detected anomaly
            both_anomalies = result_data[
                result_data['discount_anomaly_zscore'] & result_data['discount_anomaly_iqr']
            ]
            result_data.loc[both_anomalies.index, 'discount_anomaly_reason'] = 'Z-score + IQR Outlier'
            
            # Update statistics
            discount_anomaly_count = (
                result_data['discount_anomaly_zscore'] | result_data['discount_anomaly_iqr']
            ).sum()
            self.detection_stats['discount_anomalies'] += discount_anomaly_count
            
            logger.info(f"Detected {discount_anomaly_count} discount anomalies")
            
            return result_data
            
        except Exception as e:
            error_msg = f"Error detecting discount anomalies: {str(e)}"
            logger.error(error_msg)
            self.detection_stats['detection_errors'].append(error_msg)
            raise
    
    def generate_top_anomaly_records(self, data: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Generate top 5 anomaly records table.
        
        Args:
            data: DataFrame with anomaly detection results
            top_n: Number of top anomalies to return (default: 5)
            
        Returns:
            DataFrame containing top anomaly records
            
        Requirements: 3.5, 7.3
        """
        logger.info(f"Generating top {top_n} anomaly records")
        
        try:
            # Ensure we have anomaly detection results
            anomaly_columns = [
                'revenue_anomaly_zscore', 'revenue_anomaly_iqr',
                'quantity_anomaly_zscore', 'quantity_anomaly_iqr',
                'discount_anomaly_zscore', 'discount_anomaly_iqr'
            ]
            
            missing_columns = [col for col in anomaly_columns if col not in data.columns]
            if missing_columns:
                logger.warning(f"Missing anomaly detection columns: {missing_columns}")
                # Run anomaly detection if not already done
                data_with_anomalies = self.detect_all_anomalies(data)
            else:
                data_with_anomalies = data.copy()
            
            # Create overall anomaly flag
            data_with_anomalies['is_anomaly'] = (
                data_with_anomalies.get('revenue_anomaly_zscore', False) |
                data_with_anomalies.get('revenue_anomaly_iqr', False) |
                data_with_anomalies.get('quantity_anomaly_zscore', False) |
                data_with_anomalies.get('quantity_anomaly_iqr', False) |
                data_with_anomalies.get('discount_anomaly_zscore', False) |
                data_with_anomalies.get('discount_anomaly_iqr', False)
            )
            
            # Filter to anomalous records only
            anomaly_records = data_with_anomalies[data_with_anomalies['is_anomaly']].copy()
            
            if len(anomaly_records) == 0:
                logger.warning("No anomaly records found")
                return pd.DataFrame()
            
            # Calculate composite anomaly score
            anomaly_records['composite_anomaly_score'] = (
                anomaly_records.get('revenue_anomaly_score', 0) +
                anomaly_records.get('quantity_anomaly_score', 0) +
                anomaly_records.get('discount_anomaly_score', 0)
            )
            
            # Create anomaly type summary
            def get_anomaly_types(row):
                types = []
                if row.get('revenue_anomaly_zscore', False) or row.get('revenue_anomaly_iqr', False):
                    types.append('Revenue')
                if row.get('quantity_anomaly_zscore', False) or row.get('quantity_anomaly_iqr', False):
                    types.append('Quantity')
                if row.get('discount_anomaly_zscore', False) or row.get('discount_anomaly_iqr', False):
                    types.append('Discount')
                return ', '.join(types)
            
            anomaly_records['anomaly_types'] = anomaly_records.apply(get_anomaly_types, axis=1)
            
            # Create detailed anomaly reason
            def get_detailed_reason(row):
                reasons = []
                if row.get('revenue_anomaly_reason', 'Normal') != 'Normal':
                    reasons.append(f"Revenue: {row['revenue_anomaly_reason']}")
                if row.get('quantity_anomaly_reason', 'Normal') != 'Normal':
                    reasons.append(f"Quantity: {row['quantity_anomaly_reason']}")
                if row.get('discount_anomaly_reason', 'Normal') != 'Normal':
                    reasons.append(f"Discount: {row['discount_anomaly_reason']}")
                return '; '.join(reasons)
            
            anomaly_records['detailed_anomaly_reason'] = anomaly_records.apply(get_detailed_reason, axis=1)
            
            # Select top anomalies by composite score
            top_anomalies = anomaly_records.nlargest(top_n, 'composite_anomaly_score')
            
            # Select relevant columns for the final table
            output_columns = [
                'order_id', 'product_name', 'category', 'quantity', 'unit_price', 
                'discount_percent', 'revenue', 'region', 'sale_date',
                'anomaly_types', 'composite_anomaly_score', 'detailed_anomaly_reason'
            ]
            
            # Only include columns that exist in the data
            available_columns = [col for col in output_columns if col in top_anomalies.columns]
            top_anomalies_final = top_anomalies[available_columns].copy()
            
            # Add ranking
            top_anomalies_final['anomaly_rank'] = range(1, len(top_anomalies_final) + 1)
            
            # Store for reporting
            self.anomaly_records = top_anomalies_final.to_dict('records')
            
            logger.info(f"Generated top {len(top_anomalies_final)} anomaly records")
            
            return top_anomalies_final
            
        except Exception as e:
            error_msg = f"Error generating top anomaly records: {str(e)}"
            logger.error(error_msg)
            self.detection_stats['detection_errors'].append(error_msg)
            raise
    
    def detect_all_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run all anomaly detection methods on the data.
        
        Args:
            data: DataFrame containing sales data
            
        Returns:
            DataFrame with all anomaly detection results
        """
        logger.info(f"Running comprehensive anomaly detection on {len(data)} records")
        
        # Note: Individual detection methods will update total_records_analyzed
        # We don't update it here to avoid double-counting
        
        # Run revenue anomaly detection
        data_with_revenue_anomalies = self.detect_revenue_anomalies(data)
        
        # Run quantity anomaly detection
        data_with_quantity_anomalies = self.detect_quantity_anomalies(data_with_revenue_anomalies)
        
        # Run discount anomaly detection
        data_with_all_anomalies = self.detect_discount_anomalies(data_with_quantity_anomalies)
        
        # Update total anomalies count
        total_anomalies = (
            data_with_all_anomalies.get('revenue_anomaly_zscore', pd.Series([False] * len(data))) |
            data_with_all_anomalies.get('revenue_anomaly_iqr', pd.Series([False] * len(data))) |
            data_with_all_anomalies.get('quantity_anomaly_zscore', pd.Series([False] * len(data))) |
            data_with_all_anomalies.get('quantity_anomaly_iqr', pd.Series([False] * len(data))) |
            data_with_all_anomalies.get('discount_anomaly_zscore', pd.Series([False] * len(data))) |
            data_with_all_anomalies.get('discount_anomaly_iqr', pd.Series([False] * len(data)))
        ).sum()
        
        self.detection_stats['total_anomalies'] += total_anomalies
        
        logger.info(f"Comprehensive anomaly detection completed. Found {total_anomalies} total anomalies")
        
        return data_with_all_anomalies
    
    def generate_anomaly_report(self) -> Dict[str, Any]:
        """
        Generate detailed anomaly reporting with reasons.
        
        Returns:
            Dictionary containing comprehensive anomaly report
        """
        logger.info("Generating comprehensive anomaly report")
        
        report = {
            'summary': {
                'total_records_analyzed': self.detection_stats['total_records_analyzed'],
                'total_anomalies_detected': self.detection_stats['total_anomalies'],
                'anomaly_rate': (
                    self.detection_stats['total_anomalies'] / 
                    self.detection_stats['total_records_analyzed']
                    if self.detection_stats['total_records_analyzed'] > 0 else 0
                ),
                'revenue_anomalies': self.detection_stats['revenue_anomalies'],
                'quantity_anomalies': self.detection_stats['quantity_anomalies'],
                'discount_anomalies': self.detection_stats['discount_anomalies']
            },
            'detection_parameters': {
                'z_score_threshold': self.z_score_threshold,
                'iqr_multiplier': self.iqr_multiplier
            },
            'anomaly_breakdown': {
                'revenue_anomaly_rate': (
                    self.detection_stats['revenue_anomalies'] / 
                    self.detection_stats['total_records_analyzed']
                    if self.detection_stats['total_records_analyzed'] > 0 else 0
                ),
                'quantity_anomaly_rate': (
                    self.detection_stats['quantity_anomalies'] / 
                    self.detection_stats['total_records_analyzed']
                    if self.detection_stats['total_records_analyzed'] > 0 else 0
                ),
                'discount_anomaly_rate': (
                    self.detection_stats['discount_anomalies'] / 
                    self.detection_stats['total_records_analyzed']
                    if self.detection_stats['total_records_analyzed'] > 0 else 0
                )
            },
            'top_anomaly_records': self.anomaly_records,
            'detection_errors': self.detection_stats['detection_errors'],
            'recommendations': self._generate_anomaly_recommendations()
        }
        
        return report
    
    def _generate_anomaly_recommendations(self) -> List[str]:
        """Generate recommendations based on anomaly detection results."""
        recommendations = []
        
        # Overall anomaly rate recommendations
        anomaly_rate = (
            self.detection_stats['total_anomalies'] / 
            self.detection_stats['total_records_analyzed']
            if self.detection_stats['total_records_analyzed'] > 0 else 0
        )
        
        if anomaly_rate > 0.05:  # More than 5% anomalies
            recommendations.append(
                f"High anomaly rate detected ({anomaly_rate:.1%}). "
                f"Consider investigating data collection processes or adjusting detection thresholds."
            )
        elif anomaly_rate < 0.001:  # Less than 0.1% anomalies
            recommendations.append(
                f"Very low anomaly rate ({anomaly_rate:.1%}). "
                f"Consider lowering detection thresholds to catch more subtle anomalies."
            )
        
        # Field-specific recommendations
        if self.detection_stats['revenue_anomalies'] > 0:
            recommendations.append(
                f"Found {self.detection_stats['revenue_anomalies']} revenue anomalies. "
                f"Review high-value transactions for potential fraud or data entry errors."
            )
        
        if self.detection_stats['quantity_anomalies'] > 0:
            recommendations.append(
                f"Found {self.detection_stats['quantity_anomalies']} quantity anomalies. "
                f"Investigate bulk orders or potential inventory management issues."
            )
        
        if self.detection_stats['discount_anomalies'] > 0:
            recommendations.append(
                f"Found {self.detection_stats['discount_anomalies']} discount anomalies. "
                f"Review discount policies and authorization processes."
            )
        
        # Error handling recommendations
        if self.detection_stats['detection_errors']:
            recommendations.append(
                f"Encountered {len(self.detection_stats['detection_errors'])} detection errors. "
                f"Review data quality and detection algorithm robustness."
            )
        
        if not recommendations:
            recommendations.append(
                "Anomaly detection completed successfully with normal patterns detected."
            )
        
        return recommendations
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get anomaly detection statistics.
        
        Returns:
            Dictionary containing detection statistics
        """
        return {
            'total_records_analyzed': self.detection_stats['total_records_analyzed'],
            'total_anomalies_detected': self.detection_stats['total_anomalies'],
            'revenue_anomalies': self.detection_stats['revenue_anomalies'],
            'quantity_anomalies': self.detection_stats['quantity_anomalies'],
            'discount_anomalies': self.detection_stats['discount_anomalies'],
            'detection_errors': len(self.detection_stats['detection_errors']),
            'anomaly_rate': (
                self.detection_stats['total_anomalies'] / 
                self.detection_stats['total_records_analyzed']
                if self.detection_stats['total_records_analyzed'] > 0 else 0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset detection statistics."""
        self.detection_stats = {
            'total_records_analyzed': 0,
            'revenue_anomalies': 0,
            'quantity_anomalies': 0,
            'discount_anomalies': 0,
            'total_anomalies': 0,
            'detection_errors': []
        }
        self.anomaly_records = []