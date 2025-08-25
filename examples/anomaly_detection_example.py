"""
Example demonstrating the anomaly detection system.

This script shows how to use the AnomalyDetector and StatisticalAnalyzer
classes to identify suspicious transactions in e-commerce data.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from quality.anomaly_detector import AnomalyDetector
from quality.statistical_analyzer import StatisticalAnalyzer


def create_sample_data():
    """Create sample e-commerce data with known anomalies."""
    np.random.seed(42)
    
    # Generate normal data
    data = {
        'order_id': [f'ORD{i:06d}' for i in range(1000)],
        'product_name': [f'Product {i % 50}' for i in range(1000)],
        'category': np.random.choice(['Electronics', 'Fashion', 'Home'], 1000),
        'quantity': np.random.randint(1, 20, 1000),
        'unit_price': np.random.uniform(10, 1000, 1000),
        'discount_percent': np.random.uniform(0, 0.3, 1000),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'sale_date': pd.date_range('2023-01-01', periods=1000, freq='h'),
        'customer_email': [f'user{i}@example.com' for i in range(1000)]
    }
    
    df = pd.DataFrame(data)
    df['revenue'] = df['quantity'] * df['unit_price'] * (1 - df['discount_percent'])
    
    # Inject known anomalies
    print("Injecting known anomalies...")
    
    # Revenue anomalies - extremely high values
    anomaly_indices = [0, 1, 2, 3, 4]
    df.loc[anomaly_indices, 'revenue'] = [50000, 45000, 60000, 55000, 48000]
    print(f"Added {len(anomaly_indices)} revenue anomalies")
    
    # Quantity anomalies - bulk orders
    quantity_anomaly_indices = [10, 11, 12, 13, 14]
    df.loc[quantity_anomaly_indices, 'quantity'] = [500, 450, 600, 550, 480]
    print(f"Added {len(quantity_anomaly_indices)} quantity anomalies")
    
    # Discount anomalies - extreme discounts
    discount_anomaly_indices = [20, 21, 22, 23, 24]
    df.loc[discount_anomaly_indices, 'discount_percent'] = [0.9, 0.85, 0.95, 0.88, 0.92]
    print(f"Added {len(discount_anomaly_indices)} discount anomalies")
    
    # Pattern anomalies - high quantity + high discount
    pattern_anomaly_indices = [30, 31, 32]
    df.loc[pattern_anomaly_indices, 'quantity'] = [150, 200, 180]
    df.loc[pattern_anomaly_indices, 'discount_percent'] = [0.6, 0.7, 0.65]
    print(f"Added {len(pattern_anomaly_indices)} pattern anomalies")
    
    return df


def demonstrate_basic_anomaly_detection():
    """Demonstrate basic anomaly detection using AnomalyDetector."""
    print("\n" + "="*60)
    print("BASIC ANOMALY DETECTION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    data = create_sample_data()
    print(f"\nCreated sample dataset with {len(data)} records")
    
    # Initialize anomaly detector
    detector = AnomalyDetector(z_score_threshold=3.0, iqr_multiplier=1.5)
    print("Initialized AnomalyDetector with default parameters")
    
    # Run comprehensive anomaly detection
    print("\nRunning comprehensive anomaly detection...")
    data_with_anomalies = detector.detect_all_anomalies(data)
    
    # Generate top anomaly records
    print("Generating top 10 anomaly records...")
    top_anomalies = detector.generate_top_anomaly_records(data_with_anomalies, top_n=10)
    
    # Display results
    print(f"\nDetection Results:")
    stats = detector.get_detection_stats()
    print(f"- Total records analyzed: {stats['total_records_analyzed']}")
    print(f"- Total anomalies detected: {stats['total_anomalies_detected']}")
    print(f"- Anomaly rate: {stats['anomaly_rate']:.2%}")
    print(f"- Revenue anomalies: {stats['revenue_anomalies']}")
    print(f"- Quantity anomalies: {stats['quantity_anomalies']}")
    print(f"- Discount anomalies: {stats['discount_anomalies']}")
    
    # Show top anomalies
    if len(top_anomalies) > 0:
        print(f"\nTop {len(top_anomalies)} Anomalies:")
        print("-" * 80)
        for idx, row in top_anomalies.head().iterrows():
            print(f"Rank {row['anomaly_rank']}: Order {row['order_id']}")
            print(f"  Types: {row['anomaly_types']}")
            print(f"  Score: {row['composite_anomaly_score']:.2f}")
            print(f"  Revenue: ${row['revenue']:,.2f}")
            print(f"  Reason: {row['detailed_anomaly_reason']}")
            print()
    
    return data_with_anomalies, detector


def demonstrate_statistical_analysis():
    """Demonstrate advanced statistical analysis using StatisticalAnalyzer."""
    print("\n" + "="*60)
    print("ADVANCED STATISTICAL ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize statistical analyzer with high sensitivity
    analyzer = StatisticalAnalyzer(sensitivity='high', contamination=0.1)
    print("Initialized StatisticalAnalyzer with high sensitivity")
    
    # Run pattern-based anomaly detection
    print("\nRunning pattern-based anomaly detection...")
    data_with_patterns = analyzer.detect_pattern_based_anomalies(data)
    
    # Run Isolation Forest detection
    print("Running Isolation Forest multivariate anomaly detection...")
    features = ['quantity', 'unit_price', 'discount_percent', 'revenue']
    data_with_isolation = analyzer.detect_outliers_isolation_forest(data_with_patterns, features)
    
    # Generate detailed report
    print("Generating detailed anomaly report...")
    report = analyzer.generate_detailed_anomaly_report(data_with_isolation)
    
    # Display results
    print(f"\nStatistical Analysis Results:")
    summary = report['analysis_summary']
    print(f"- Total records analyzed: {summary['total_records_analyzed']}")
    print(f"- Sensitivity level: {summary['sensitivity_level']}")
    
    if 'anomaly_statistics' in report:
        stats = report['anomaly_statistics']
        for anomaly_type, info in stats.items():
            print(f"- {anomaly_type}: {info['count']} ({info['percentage']:.1f}%)")
    
    # Show pattern analysis
    if 'pattern_analysis' in report and 'common_patterns' in report['pattern_analysis']:
        print(f"\nCommon Anomaly Patterns:")
        patterns = report['pattern_analysis']['common_patterns']
        for pattern, count in list(patterns.items())[:5]:
            print(f"- {pattern}: {count} occurrences")
    
    # Show recommendations
    if 'recommendations' in report:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"{i}. {rec}")
    
    return data_with_isolation, analyzer


def demonstrate_threshold_calculation():
    """Demonstrate statistical threshold calculation."""
    print("\n" + "="*60)
    print("STATISTICAL THRESHOLD CALCULATION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(sensitivity='medium')
    
    # Calculate thresholds for different fields
    fields_to_analyze = ['revenue', 'quantity', 'unit_price', 'discount_percent']
    
    for field in fields_to_analyze:
        print(f"\nThresholds for {field}:")
        print("-" * 40)
        
        thresholds = analyzer.calculate_statistical_thresholds(data[field], method='all')
        
        # Display key thresholds
        if 'zscore_upper_threshold' in thresholds:
            print(f"Z-score upper threshold: {thresholds['zscore_upper_threshold']:.2f}")
        if 'iqr_upper_threshold' in thresholds:
            print(f"IQR upper threshold: {thresholds['iqr_upper_threshold']:.2f}")
        if 'percentile_upper' in thresholds:
            print(f"95th percentile: {thresholds['percentile_upper']:.2f}")
        
        # Show basic statistics
        print(f"Mean: {data[field].mean():.2f}")
        print(f"Std: {data[field].std():.2f}")
        print(f"Median: {data[field].median():.2f}")


def main():
    """Main demonstration function."""
    print("ANOMALY DETECTION SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("This example demonstrates the anomaly detection capabilities")
    print("of the scalable data pipeline system.")
    
    try:
        # Run demonstrations
        data_with_anomalies, detector = demonstrate_basic_anomaly_detection()
        data_with_statistical, analyzer = demonstrate_statistical_analysis()
        demonstrate_threshold_calculation()
        
        # Generate final report
        print("\n" + "="*60)
        print("FINAL ANOMALY REPORT")
        print("="*60)
        
        final_report = detector.generate_anomaly_report()
        
        print(f"\nOverall Summary:")
        summary = final_report['summary']
        print(f"- Records processed: {summary['total_records_analyzed']}")
        print(f"- Anomalies found: {summary['total_anomalies_detected']}")
        print(f"- Detection rate: {summary['anomaly_rate']:.2%}")
        
        print(f"\nDetection Parameters:")
        params = final_report['detection_parameters']
        print(f"- Z-score threshold: {params['z_score_threshold']}")
        print(f"- IQR multiplier: {params['iqr_multiplier']}")
        
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(final_report['recommendations'][:3], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()