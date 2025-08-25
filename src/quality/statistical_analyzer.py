"""
Statistical analyzer for advanced anomaly detection.

This module provides the StatisticalAnalyzer class that implements advanced
statistical methods for anomaly detection with configurable sensitivity
parameters and detailed reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Advanced statistical analyzer for anomaly detection.
    
    This class provides sophisticated statistical methods for anomaly detection
    including configurable sensitivity parameters, pattern-based detection,
    and detailed anomaly reporting with reasons.
    """
    
    def __init__(self, 
                 sensitivity: str = 'medium',
                 contamination: float = 0.1,
                 random_state: int = 42):
        """
        Initialize the StatisticalAnalyzer.
        
        Args:
            sensitivity: Detection sensitivity ('low', 'medium', 'high')
            contamination: Expected proportion of anomalies (for Isolation Forest)
            random_state: Random state for reproducible results
        """
        self.sensitivity = sensitivity
        self.contamination = contamination
        self.random_state = random_state
        
        # Configure sensitivity parameters
        self.sensitivity_params = self._configure_sensitivity_parameters()
        
        self.analysis_stats = {
            'total_records_analyzed': 0,
            'statistical_anomalies': 0,
            'pattern_anomalies': 0,
            'isolation_forest_anomalies': 0,
            'analysis_errors': []
        }
        
        self.threshold_cache = {}
    
    def _configure_sensitivity_parameters(self) -> Dict[str, Any]:
        """
        Configure sensitivity parameters based on sensitivity level.
        
        Returns:
            Dictionary containing sensitivity parameters
        """
        sensitivity_configs = {
            'low': {
                'z_score_threshold': 3.5,
                'iqr_multiplier': 2.0,
                'percentile_threshold': 0.99,
                'isolation_forest_contamination': 0.05
            },
            'medium': {
                'z_score_threshold': 3.0,
                'iqr_multiplier': 1.5,
                'percentile_threshold': 0.95,
                'isolation_forest_contamination': 0.1
            },
            'high': {
                'z_score_threshold': 2.5,
                'iqr_multiplier': 1.2,
                'percentile_threshold': 0.90,
                'isolation_forest_contamination': 0.15
            }
        }
        
        return sensitivity_configs.get(self.sensitivity, sensitivity_configs['medium'])
    
    def calculate_statistical_thresholds(self, data: pd.Series, method: str = 'all') -> Dict[str, float]:
        """
        Create statistical threshold calculation methods.
        
        Args:
            data: Series containing numerical data
            method: Statistical method ('zscore', 'iqr', 'percentile', 'all')
            
        Returns:
            Dictionary containing calculated thresholds
            
        Requirements: 7.4
        """
        logger.info(f"Calculating statistical thresholds for {len(data)} values using {method} method")
        
        try:
            # Remove null values first
            clean_data = data.dropna()
            
            if len(clean_data) == 0:
                logger.warning("No valid data for threshold calculation")
                return {}
            
            # Remove infinite values only for numeric data
            if pd.api.types.is_numeric_dtype(clean_data):
                clean_data = clean_data[np.isfinite(clean_data)]
                
                if len(clean_data) == 0:
                    logger.warning("No finite numeric data for threshold calculation")
                    return {}
            
            thresholds = {}
            
            # Z-score based thresholds
            if method in ['zscore', 'all']:
                mean_val = float(clean_data.mean())
                std_val = float(clean_data.std())
                
                if std_val > 0:
                    z_threshold = self.sensitivity_params['z_score_threshold']
                    thresholds.update({
                        'zscore_mean': mean_val,
                        'zscore_std': std_val,
                        'zscore_upper_threshold': mean_val + (z_threshold * std_val),
                        'zscore_lower_threshold': mean_val - (z_threshold * std_val),
                        'zscore_threshold_value': z_threshold
                    })
                else:
                    thresholds.update({
                        'zscore_mean': mean_val,
                        'zscore_std': 0.0,
                        'zscore_upper_threshold': mean_val,
                        'zscore_lower_threshold': mean_val,
                        'zscore_threshold_value': 0.0
                    })
            
            # IQR based thresholds
            if method in ['iqr', 'all']:
                q1 = float(clean_data.quantile(0.25))
                q3 = float(clean_data.quantile(0.75))
                iqr = q3 - q1
                
                iqr_multiplier = self.sensitivity_params['iqr_multiplier']
                thresholds.update({
                    'iqr_q1': q1,
                    'iqr_q3': q3,
                    'iqr_value': iqr,
                    'iqr_upper_threshold': q3 + (iqr_multiplier * iqr),
                    'iqr_lower_threshold': q1 - (iqr_multiplier * iqr),
                    'iqr_multiplier': iqr_multiplier
                })
            
            # Percentile based thresholds
            if method in ['percentile', 'all']:
                percentile_threshold = self.sensitivity_params['percentile_threshold']
                lower_percentile = (1 - percentile_threshold) / 2
                upper_percentile = percentile_threshold + lower_percentile
                
                thresholds.update({
                    'percentile_lower': float(clean_data.quantile(lower_percentile)),
                    'percentile_upper': float(clean_data.quantile(upper_percentile)),
                    'percentile_threshold': percentile_threshold
                })
            
            # Modified Z-score (using median absolute deviation)
            if method in ['modified_zscore', 'all']:
                median_val = float(clean_data.median())
                mad = float(np.median(np.abs(clean_data - median_val)))
                
                if mad > 0:
                    modified_z_threshold = 3.5  # Standard threshold for modified z-score
                    thresholds.update({
                        'modified_zscore_median': median_val,
                        'modified_zscore_mad': mad,
                        'modified_zscore_threshold': modified_z_threshold,
                        'modified_zscore_upper': median_val + (modified_z_threshold * mad * 1.4826),
                        'modified_zscore_lower': median_val - (modified_z_threshold * mad * 1.4826)
                    })
            
            # Cache thresholds for reuse
            cache_key = f"{data.name}_{method}_{self.sensitivity}"
            self.threshold_cache[cache_key] = thresholds
            
            logger.info(f"Calculated {len(thresholds)} threshold values")
            return thresholds
            
        except Exception as e:
            error_msg = f"Error calculating statistical thresholds: {str(e)}"
            logger.error(error_msg)
            self.analysis_stats['analysis_errors'].append(error_msg)
            raise
    
    def detect_outliers_isolation_forest(self, data: pd.DataFrame, 
                                       features: List[str]) -> pd.DataFrame:
        """
        Implement Isolation Forest for multivariate anomaly detection.
        
        Args:
            data: DataFrame containing features for analysis
            features: List of feature column names to analyze
            
        Returns:
            DataFrame with isolation forest anomaly results
        """
        logger.info(f"Running Isolation Forest anomaly detection on {len(features)} features")
        
        try:
            # Ensure all features exist
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            result_data = data.copy()
            
            # Extract feature data and handle missing values
            feature_data = data[features].copy()
            
            # Remove rows with any missing values in features
            complete_rows = feature_data.dropna()
            
            if len(complete_rows) == 0:
                logger.warning("No complete rows for Isolation Forest analysis")
                result_data['isolation_forest_anomaly'] = False
                result_data['isolation_forest_score'] = 0.0
                return result_data
            
            # Scale features for better performance
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(complete_rows)
            
            # Configure Isolation Forest
            contamination = self.sensitivity_params['isolation_forest_contamination']
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=self.random_state,
                n_estimators=100
            )
            
            # Fit and predict
            anomaly_labels = iso_forest.fit_predict(scaled_features)
            anomaly_scores = iso_forest.decision_function(scaled_features)
            
            # Initialize results for all records
            result_data['isolation_forest_anomaly'] = False
            result_data['isolation_forest_score'] = 0.0
            
            # Set results for complete rows
            result_data.loc[complete_rows.index, 'isolation_forest_anomaly'] = (anomaly_labels == -1)
            result_data.loc[complete_rows.index, 'isolation_forest_score'] = anomaly_scores
            
            # Update statistics
            anomaly_count = (anomaly_labels == -1).sum()
            self.analysis_stats['isolation_forest_anomalies'] += anomaly_count
            
            logger.info(f"Isolation Forest detected {anomaly_count} anomalies")
            
            return result_data
            
        except Exception as e:
            error_msg = f"Error in Isolation Forest detection: {str(e)}"
            logger.error(error_msg)
            self.analysis_stats['analysis_errors'].append(error_msg)
            raise
    
    def detect_pattern_based_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build pattern-based anomaly detection for suspicious combinations.
        
        Args:
            data: DataFrame containing sales data
            
        Returns:
            DataFrame with pattern-based anomaly flags
            
        Requirements: 7.5
        """
        logger.info(f"Detecting pattern-based anomalies in {len(data)} records")
        
        # Update total records analyzed
        self.analysis_stats['total_records_analyzed'] += len(data)
        
        try:
            result_data = data.copy()
            
            # Initialize pattern anomaly columns
            result_data['pattern_anomaly'] = False
            result_data['pattern_anomaly_reasons'] = ''
            result_data['pattern_anomaly_score'] = 0.0
            
            anomaly_reasons = []
            pattern_scores = []
            
            for idx, row in result_data.iterrows():
                reasons = []
                score = 0.0
                
                # Pattern 1: High quantity with high discount (potential bulk fraud)
                if (row.get('quantity', 0) > 100 and 
                    row.get('discount_percent', 0) > 0.5):
                    reasons.append('High quantity + High discount')
                    score += 2.0
                
                # Pattern 2: Very high unit price with very high discount
                if (row.get('unit_price', 0) > 10000 and 
                    row.get('discount_percent', 0) > 0.7):
                    reasons.append('High price + Extreme discount')
                    score += 3.0
                
                # Pattern 3: Zero or very low unit price with high quantity
                if (row.get('unit_price', 0) < 1.0 and 
                    row.get('quantity', 0) > 50):
                    reasons.append('Low price + High quantity')
                    score += 1.5
                
                # Pattern 4: Weekend sales with unusually high values
                if 'sale_date' in row and pd.notna(row['sale_date']):
                    try:
                        sale_date = pd.to_datetime(row['sale_date'])
                        if (sale_date.weekday() >= 5 and  # Weekend
                            row.get('revenue', 0) > 50000):
                            reasons.append('Weekend high-value sale')
                            score += 1.0
                    except:
                        pass
                
                # Pattern 5: Suspicious email patterns (if available)
                if 'customer_email' in row and pd.notna(row['customer_email']):
                    email = str(row['customer_email']).lower()
                    # Check for suspicious patterns
                    if (any(char * 3 in email for char in 'abcdefghijklmnopqrstuvwxyz') or
                        email.count('.') > 3 or
                        len(email.split('@')[0]) < 3):
                        reasons.append('Suspicious email pattern')
                        score += 0.5
                
                # Pattern 6: Round number combinations (potential manual entry fraud)
                if (row.get('quantity', 0) % 10 == 0 and 
                    row.get('unit_price', 0) % 100 == 0 and
                    row.get('quantity', 0) > 10 and
                    row.get('unit_price', 0) > 100):
                    reasons.append('Round number combination')
                    score += 0.8
                
                # Pattern 7: Inconsistent regional pricing
                if 'region' in row and 'category' in row:
                    # This would require regional price analysis - simplified version
                    if (row.get('unit_price', 0) > 100000):  # Very high price
                        reasons.append('Extreme unit price')
                        score += 1.2
                
                # Pattern 8: Duplicate-like patterns in product names
                if 'product_name' in row and pd.notna(row['product_name']):
                    product_name = str(row['product_name']).lower()
                    # Check for repeated characters or suspicious patterns
                    if (len(set(product_name.replace(' ', ''))) < 5 and len(product_name) > 10):
                        reasons.append('Suspicious product name pattern')
                        score += 0.3
                
                anomaly_reasons.append('; '.join(reasons) if reasons else '')
                pattern_scores.append(score)
            
            # Set pattern anomaly results
            result_data['pattern_anomaly_reasons'] = anomaly_reasons
            result_data['pattern_anomaly_score'] = pattern_scores
            
            # Define threshold for pattern anomalies based on sensitivity
            score_thresholds = {
                'low': 3.0,
                'medium': 2.0,
                'high': 1.0
            }
            
            threshold = score_thresholds.get(self.sensitivity, 2.0)
            result_data['pattern_anomaly'] = result_data['pattern_anomaly_score'] >= threshold
            
            # Update statistics
            pattern_anomaly_count = result_data['pattern_anomaly'].sum()
            self.analysis_stats['pattern_anomalies'] += pattern_anomaly_count
            
            logger.info(f"Detected {pattern_anomaly_count} pattern-based anomalies")
            
            return result_data
            
        except Exception as e:
            error_msg = f"Error in pattern-based anomaly detection: {str(e)}"
            logger.error(error_msg)
            self.analysis_stats['analysis_errors'].append(error_msg)
            raise
    
    def generate_detailed_anomaly_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Add detailed anomaly reporting with reasons.
        
        Args:
            data: DataFrame with anomaly detection results
            
        Returns:
            Dictionary containing detailed anomaly report
            
        Requirements: 7.4
        """
        logger.info("Generating detailed anomaly report")
        
        try:
            # Update total records analyzed
            self.analysis_stats['total_records_analyzed'] += len(data)
            
            # Analyze anomaly patterns
            anomaly_columns = [col for col in data.columns if 'anomaly' in col.lower()]
            
            report = {
                'analysis_summary': {
                    'total_records_analyzed': len(data),
                    'sensitivity_level': self.sensitivity,
                    'contamination_rate': self.contamination,
                    'analysis_parameters': self.sensitivity_params
                },
                'anomaly_statistics': {},
                'threshold_analysis': {},
                'pattern_analysis': {},
                'recommendations': [],
                'detailed_findings': []
            }
            
            # Statistical anomaly analysis
            if any('statistical' in col for col in anomaly_columns):
                statistical_anomalies = data[data.get('statistical_anomaly', pd.Series([False] * len(data)))]
                report['anomaly_statistics']['statistical_anomalies'] = {
                    'count': len(statistical_anomalies),
                    'percentage': len(statistical_anomalies) / len(data) * 100 if len(data) > 0 else 0
                }
            
            # Pattern anomaly analysis
            if 'pattern_anomaly' in data.columns:
                pattern_anomalies = data[data['pattern_anomaly']]
                report['anomaly_statistics']['pattern_anomalies'] = {
                    'count': len(pattern_anomalies),
                    'percentage': len(pattern_anomalies) / len(data) * 100 if len(data) > 0 else 0
                }
                
                # Analyze pattern reasons
                if 'pattern_anomaly_reasons' in data.columns:
                    reason_counts = {}
                    for reasons in pattern_anomalies['pattern_anomaly_reasons']:
                        if reasons:
                            for reason in reasons.split('; '):
                                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    
                    report['pattern_analysis']['common_patterns'] = dict(
                        sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
                    )
            
            # Isolation Forest analysis
            if 'isolation_forest_anomaly' in data.columns:
                iso_anomalies = data[data['isolation_forest_anomaly']]
                report['anomaly_statistics']['isolation_forest_anomalies'] = {
                    'count': len(iso_anomalies),
                    'percentage': len(iso_anomalies) / len(data) * 100 if len(data) > 0 else 0
                }
            
            # Threshold analysis for numerical columns
            numerical_columns = data.select_dtypes(include=[np.number]).columns
            for col in numerical_columns:
                if col in data.columns and not col.endswith('_anomaly') and not col.endswith('_score'):
                    try:
                        thresholds = self.calculate_statistical_thresholds(data[col])
                        report['threshold_analysis'][col] = thresholds
                    except Exception as e:
                        logger.warning(f"Could not calculate thresholds for {col}: {str(e)}")
            
            # Generate recommendations
            report['recommendations'] = self._generate_detailed_recommendations(data, report)
            
            # Detailed findings for top anomalies
            if len(data) > 0:
                # Find records with multiple anomaly types
                anomaly_score_cols = [col for col in data.columns if col.endswith('_score') and 'anomaly' in col]
                if anomaly_score_cols:
                    data_copy = data.copy()
                    data_copy['total_anomaly_score'] = data_copy[anomaly_score_cols].sum(axis=1)
                    top_anomalies = data_copy.nlargest(5, 'total_anomaly_score')
                    
                    for idx, row in top_anomalies.iterrows():
                        finding = {
                            'record_id': idx,
                            'total_anomaly_score': row.get('total_anomaly_score', 0),
                            'anomaly_types': [],
                            'key_values': {}
                        }
                        
                        # Identify which anomaly types are present
                        for col in data.columns:
                            if col.endswith('_anomaly') and row.get(col, False):
                                finding['anomaly_types'].append(col.replace('_anomaly', ''))
                        
                        # Extract key values
                        key_fields = ['order_id', 'revenue', 'quantity', 'unit_price', 'discount_percent']
                        for field in key_fields:
                            if field in row:
                                finding['key_values'][field] = row[field]
                        
                        report['detailed_findings'].append(finding)
            
            return report
            
        except Exception as e:
            error_msg = f"Error generating detailed anomaly report: {str(e)}"
            logger.error(error_msg)
            self.analysis_stats['analysis_errors'].append(error_msg)
            raise
    
    def _generate_detailed_recommendations(self, data: pd.DataFrame, report: Dict[str, Any]) -> List[str]:
        """Generate detailed recommendations based on analysis results."""
        recommendations = []
        
        # Sensitivity recommendations
        total_anomalies = sum([
            stats.get('count', 0) for stats in report['anomaly_statistics'].values()
        ])
        anomaly_rate = total_anomalies / len(data) if len(data) > 0 else 0
        
        if anomaly_rate > 0.2:  # More than 20% anomalies
            recommendations.append(
                f"Very high anomaly rate ({anomaly_rate:.1%}) detected. "
                f"Consider reducing sensitivity or investigating data quality issues."
            )
        elif anomaly_rate < 0.01:  # Less than 1% anomalies
            recommendations.append(
                f"Low anomaly rate ({anomaly_rate:.1%}) detected. "
                f"Consider increasing sensitivity to catch more subtle patterns."
            )
        
        # Pattern-specific recommendations
        if 'pattern_analysis' in report and 'common_patterns' in report['pattern_analysis']:
            common_patterns = report['pattern_analysis']['common_patterns']
            
            if 'High quantity + High discount' in common_patterns:
                recommendations.append(
                    "Detected bulk discount anomalies. Review bulk pricing policies and authorization levels."
                )
            
            if 'High price + Extreme discount' in common_patterns:
                recommendations.append(
                    "Detected extreme discount patterns on high-value items. "
                    "Implement stricter approval processes for large discounts."
                )
            
            if 'Suspicious email pattern' in common_patterns:
                recommendations.append(
                    "Detected suspicious email patterns. Implement email validation and verification processes."
                )
        
        # Statistical threshold recommendations
        if 'threshold_analysis' in report:
            for field, thresholds in report['threshold_analysis'].items():
                if 'zscore_std' in thresholds and thresholds['zscore_std'] == 0:
                    recommendations.append(
                        f"Field '{field}' has zero variance. Check for data quality issues or constant values."
                    )
        
        # Error handling recommendations
        if self.analysis_stats['analysis_errors']:
            recommendations.append(
                f"Encountered {len(self.analysis_stats['analysis_errors'])} analysis errors. "
                f"Review data quality and algorithm robustness."
            )
        
        if not recommendations:
            recommendations.append(
                "Statistical analysis completed successfully. Continue monitoring with current parameters."
            )
        
        return recommendations
    
    def configure_sensitivity(self, sensitivity: str) -> None:
        """
        Implement configurable sensitivity parameters.
        
        Args:
            sensitivity: New sensitivity level ('low', 'medium', 'high')
            
        Requirements: 7.5
        """
        if sensitivity not in ['low', 'medium', 'high']:
            raise ValueError("Sensitivity must be 'low', 'medium', or 'high'")
        
        self.sensitivity = sensitivity
        self.sensitivity_params = self._configure_sensitivity_parameters()
        
        # Clear threshold cache as parameters have changed
        self.threshold_cache.clear()
        
        logger.info(f"Sensitivity configured to {sensitivity}")
        logger.info(f"New parameters: {self.sensitivity_params}")
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis statistics.
        
        Returns:
            Dictionary containing analysis statistics
        """
        return {
            'total_records_analyzed': self.analysis_stats['total_records_analyzed'],
            'statistical_anomalies': self.analysis_stats['statistical_anomalies'],
            'pattern_anomalies': self.analysis_stats['pattern_anomalies'],
            'isolation_forest_anomalies': self.analysis_stats['isolation_forest_anomalies'],
            'analysis_errors': len(self.analysis_stats['analysis_errors']),
            'sensitivity_level': self.sensitivity,
            'sensitivity_parameters': self.sensitivity_params
        }
    
    def reset_stats(self) -> None:
        """Reset analysis statistics."""
        self.analysis_stats = {
            'total_records_analyzed': 0,
            'statistical_anomalies': 0,
            'pattern_anomalies': 0,
            'isolation_forest_anomalies': 0,
            'analysis_errors': []
        }
        self.threshold_cache.clear()