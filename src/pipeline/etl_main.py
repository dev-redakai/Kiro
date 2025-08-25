"""
Main ETL pipeline orchestrator for coordinating the entire data processing workflow.

This module provides the ETLMain class that orchestrates the complete pipeline
including data ingestion, cleaning, transformation, anomaly detection, and output.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import ConfigManager, PipelineConfig
from src.utils.logger import get_module_logger
from src.utils.error_handler import ErrorHandler
from src.utils.memory_manager import MemoryManager
from src.utils.performance_manager import PerformanceManager
from src.utils.pipeline_health import pipeline_health_monitor, start_health_monitoring, stop_health_monitoring
from src.utils.pipeline_recovery import pipeline_recovery_manager, create_checkpoint, handle_failure, FailureType
from src.utils.pipeline_alerting import pipeline_alerting_system, start_alerting, stop_alerting
from src.quality.automated_validation import automated_validator, validate_data
from src.pipeline.ingestion import DataIngestionManager
from src.data.data_cleaner import DataCleaner
from src.pipeline.transformation import AnalyticalTransformer
from src.quality.anomaly_detector import AnomalyDetector
from src.data.data_exporter import DataExporter


class PipelineStatus:
    """Pipeline execution status tracking."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.current_stage = "Not Started"
        self.stages_completed = []
        self.total_records_processed = 0
        self.errors = []
        self.warnings = []
        self.performance_metrics = {}
    
    def start_stage(self, stage_name: str):
        """Start a new pipeline stage."""
        self.current_stage = stage_name
        self.performance_metrics[stage_name] = {'start_time': time.time()}
    
    def complete_stage(self, stage_name: str, records_processed: int = 0):
        """Complete a pipeline stage."""
        if stage_name in self.performance_metrics:
            self.performance_metrics[stage_name]['end_time'] = time.time()
            self.performance_metrics[stage_name]['duration'] = (
                self.performance_metrics[stage_name]['end_time'] - 
                self.performance_metrics[stage_name]['start_time']
            )
            self.performance_metrics[stage_name]['records_processed'] = records_processed
        
        self.stages_completed.append(stage_name)
        self.total_records_processed += records_processed
    
    def add_error(self, error: str):
        """Add an error to the status."""
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Add a warning to the status."""
        self.warnings.append(warning)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary."""
        duration = 0
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
        
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': duration,
            'current_stage': self.current_stage,
            'stages_completed': self.stages_completed,
            'total_records_processed': self.total_records_processed,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'performance_metrics': self.performance_metrics
        }


class ETLMain:
    """
    Main ETL pipeline orchestrator.
    
    This class coordinates the entire data processing workflow including:
    - Data ingestion with chunked reading
    - Data cleaning and validation
    - Analytical transformations
    - Anomaly detection
    - Data export and persistence
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ETL pipeline orchestrator.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Initialize logging
        self.logger = get_module_logger("etl_main")
        
        # Initialize status tracking
        self.status = PipelineStatus()
        
        # Initialize components
        self.error_handler = ErrorHandler()
        self.memory_manager = MemoryManager(max_memory_usage=self.config.max_memory_usage)
        self.performance_manager = PerformanceManager()
        
        # Initialize monitoring and health systems
        self.health_monitoring_enabled = True
        self.recovery_enabled = True
        self.alerting_enabled = True
        self.automated_validation_enabled = True
        
        # Initialize enhanced validation configuration
        self._configure_enhanced_validation()
        
        # Pipeline components (initialized lazily)
        self._ingestion_manager = None
        self._data_cleaner = None
        self._transformer = None
        self._anomaly_detector = None
        self._data_exporter = None
        
        # Data storage
        self.processed_data = None
        self.analytical_tables = {}
        
        self.logger.info("ETL Pipeline orchestrator initialized")
    
    def _configure_enhanced_validation(self) -> None:
        """Configure enhanced validation with error handling improvements."""
        try:
            # Get validation configuration from config
            validation_config = getattr(self.config, 'validation_config', {})
            
            # Configure error resilience
            error_resilience = validation_config.get('error_resilience', {})
            if error_resilience.get('enabled', True):
                self.logger.info("Enhanced validation error resilience enabled")
                
                # Configure automated validator with enhanced settings
                if hasattr(automated_validator, 'configure_error_resilience'):
                    automated_validator.configure_error_resilience(
                        continue_on_failure=error_resilience.get('continue_on_failure', True),
                        max_error_rate=error_resilience.get('max_error_rate', 0.20),
                        isolation_mode=error_resilience.get('isolation_mode', True)
                    )
            
            # Configure error handling
            error_handling = validation_config.get('error_handling', {})
            if error_handling.get('categorization', True):
                self.logger.info("Enhanced validation error categorization enabled")
            
            if error_handling.get('detailed_logging', True):
                self.logger.info("Enhanced validation detailed logging enabled")
            
            # Configure reporting
            reporting = validation_config.get('reporting', {})
            if reporting.get('generate_error_reports', True):
                self.logger.info("Enhanced validation error reporting enabled")
            
            # Store validation configuration for later use
            self.validation_config = validation_config
            
        except Exception as e:
            self.logger.warning(f"Failed to configure enhanced validation: {e}")
            # Continue with default validation settings
            self.validation_config = {}
    
    @property
    def ingestion_manager(self) -> DataIngestionManager:
        """Lazy initialization of ingestion manager."""
        if self._ingestion_manager is None:
            config_dict = {
                'chunk_size': self.config.chunk_size,
                'encoding': 'utf-8',
                'max_retries': 3,
                'supported_formats': ['.csv', '.tsv', '.txt']
            }
            self._ingestion_manager = DataIngestionManager(config=config_dict)
        return self._ingestion_manager
    
    @property
    def data_cleaner(self) -> DataCleaner:
        """Lazy initialization of data cleaner."""
        if self._data_cleaner is None:
            self._data_cleaner = DataCleaner()
        return self._data_cleaner
    
    @property
    def transformer(self) -> AnalyticalTransformer:
        """Lazy initialization of analytical transformer."""
        if self._transformer is None:
            self._transformer = AnalyticalTransformer()
        return self._transformer
    
    @property
    def anomaly_detector(self) -> AnomalyDetector:
        """Lazy initialization of anomaly detector."""
        if self._anomaly_detector is None:
            self._anomaly_detector = AnomalyDetector(
                z_score_threshold=self.config.anomaly_threshold
            )
        return self._anomaly_detector
    
    @property
    def data_exporter(self) -> DataExporter:
        """Lazy initialization of data exporter."""
        if self._data_exporter is None:
            self._data_exporter = DataExporter(config=self.config)
        return self._data_exporter
    
    def validate_configuration(self) -> bool:
        """
        Validate pipeline configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        self.logger.info("Validating pipeline configuration")
        
        errors = self.config_manager.validate_config()
        if errors:
            for error in errors:
                self.logger.error(f"Configuration error: {error}")
                self.status.add_error(f"Configuration error: {error}")
            return False
        
        # Validate data directories exist
        data_paths = self.config.get_data_paths()
        for path_name, path_value in data_paths.items():
            path_obj = Path(path_value)
            if not path_obj.exists():
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created directory: {path_value}")
                except Exception as e:
                    error_msg = f"Failed to create {path_name} directory {path_value}: {e}"
                    self.logger.error(error_msg)
                    self.status.add_error(error_msg)
                    return False
        
        self.logger.info("Configuration validation completed successfully")
        return True
    
    def run_pipeline(self, input_file: str, skip_stages: Optional[List[str]] = None) -> bool:
        """
        Run the complete ETL pipeline.
        
        Args:
            input_file: Path to input CSV file
            skip_stages: List of stages to skip (optional)
            
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        skip_stages = skip_stages or []
        self.status.start_time = time.time()
        
        try:
            self.logger.info(f"Starting ETL pipeline for file: {input_file}")
            
            # Start monitoring systems
            self._start_monitoring_systems()
            
            # Stage 1: Configuration Validation
            if "validation" not in skip_stages:
                self.status.start_stage("Configuration Validation")
                if not self.validate_configuration():
                    return False
                self.status.complete_stage("Configuration Validation")
            
            # Stage 2: Data Ingestion
            if "ingestion" not in skip_stages:
                if not self._run_ingestion_stage(input_file):
                    return False
            
            # Stage 3: Data Cleaning
            if "cleaning" not in skip_stages:
                if not self._run_cleaning_stage():
                    return False
            
            # Stage 4: Data Transformation
            if "transformation" not in skip_stages:
                if not self._run_transformation_stage():
                    return False
            
            # Stage 5: Anomaly Detection
            if "anomaly_detection" not in skip_stages:
                if not self._run_anomaly_detection_stage():
                    return False
            
            # Stage 6: Data Export
            if "export" not in skip_stages:
                if not self._run_export_stage():
                    return False
            
            self.status.end_time = time.time()
            self.status.current_stage = "Completed"
            
            self.logger.info("ETL pipeline completed successfully")
            self._log_pipeline_summary()
            
            return True
            
        except Exception as e:
            self.status.end_time = time.time()
            self.status.current_stage = "Failed"
            error_msg = f"Pipeline failed with error: {str(e)}"
            self.logger.error(error_msg)
            self.status.add_error(error_msg)
            
            # Attempt recovery
            if self.recovery_enabled:
                self._attempt_pipeline_recovery(e, error_msg)
            
            return False
        
        finally:
            # Stop monitoring systems
            self._stop_monitoring_systems()
    
    def _run_ingestion_stage(self, input_file: str) -> bool:
        """Run data ingestion stage."""
        self.status.start_stage("Data Ingestion")
        self.logger.info("Starting data ingestion stage")
        
        try:
            # Validate input file
            if not Path(input_file).exists():
                error_msg = f"Input file not found: {input_file}"
                self.logger.error(error_msg)
                self.status.add_error(error_msg)
                return False
            
            # Process data in chunks
            processed_chunks = []
            total_records = 0
            
            for chunk_num, chunk in enumerate(self.ingestion_manager.ingest_data(input_file)):
                self.logger.info(f"Processing chunk {chunk_num + 1}")
                
                # Memory check
                if not self.memory_manager.check_memory_usage():
                    warning_msg = "High memory usage detected during ingestion"
                    self.logger.warning(warning_msg)
                    self.status.add_warning(warning_msg)
                
                processed_chunks.append(chunk)
                total_records += len(chunk)
                
                # Progress logging
                if chunk_num % 10 == 0:
                    self.logger.info(f"Processed {total_records} records so far")
            
            # Combine chunks
            if processed_chunks:
                self.processed_data = pd.concat(processed_chunks, ignore_index=True)
                self.logger.info(f"Ingestion completed: {len(self.processed_data)} total records")
            else:
                error_msg = "No data was ingested"
                self.logger.error(error_msg)
                self.status.add_error(error_msg)
                return False
            
            self.status.complete_stage("Data Ingestion", total_records)
            return True
            
        except Exception as e:
            error_msg = f"Data ingestion failed: {str(e)}"
            self.logger.error(error_msg)
            self.status.add_error(error_msg)
            return False
    
    def _run_cleaning_stage(self) -> bool:
        """Run data cleaning stage."""
        self.status.start_stage("Data Cleaning")
        self.logger.info("Starting data cleaning stage")
        
        try:
            if self.processed_data is None:
                error_msg = "No data available for cleaning"
                self.logger.error(error_msg)
                self.status.add_error(error_msg)
                return False
            
            # Create checkpoint before cleaning
            if self.recovery_enabled:
                checkpoint_id = create_checkpoint(
                    stage_name="Data Cleaning",
                    data_state={'data_shape': self.processed_data.shape},
                    processed_records=len(self.processed_data),
                    stage_progress=0.0,
                    metadata={'stage': 'pre_cleaning'}
                )
                self.logger.info(f"Created pre-cleaning checkpoint: {checkpoint_id}")
            
            # Clean data in chunks to manage memory
            chunk_size = self.config.chunk_size
            cleaned_chunks = []
            total_records = len(self.processed_data)
            
            for i in range(0, total_records, chunk_size):
                chunk = self.processed_data.iloc[i:i + chunk_size].copy()
                cleaned_chunk = self.data_cleaner.clean_chunk(chunk)
                cleaned_chunks.append(cleaned_chunk)
                
                # Progress logging
                progress = min(i + chunk_size, total_records)
                if i % (chunk_size * 10) == 0:
                    self.logger.info(f"Cleaned {progress}/{total_records} records")
                    
                    # Create intermediate checkpoint
                    if self.recovery_enabled and i % (chunk_size * 50) == 0:  # Every 50 chunks
                        stage_progress = (progress / total_records) * 100
                        create_checkpoint(
                            stage_name="Data Cleaning",
                            data_state={'cleaned_records': progress},
                            processed_records=progress,
                            stage_progress=stage_progress,
                            metadata={'stage': 'cleaning_in_progress'}
                        )
            
            # Combine cleaned chunks
            self.processed_data = pd.concat(cleaned_chunks, ignore_index=True)
            
            # Run enhanced automated data validation if enabled
            if self.automated_validation_enabled:
                self.logger.info("Running enhanced automated data quality validation with error resilience")
                
                # Use enhanced validation with error handling improvements
                validation_result = automated_validator.validate_dataframe(self.processed_data)
                
                # Log comprehensive validation results
                self.logger.info(f"Validation completed with status: {validation_result['status']}")
                
                # Handle validation results based on enhanced error handling
                if validation_result['status'] == 'critical':
                    self.logger.error(f"Critical validation failures detected: {validation_result.get('critical_failures', 0)}")
                    # Log critical issues but continue processing due to error resilience
                    for error_report in automated_validator.error_reports:
                        if error_report.error_type in ['validation_function_failure', 'error_handler_failure']:
                            self.logger.error(f"Critical validation error: {error_report.rule_name} - {error_report.error_message}")
                
                elif validation_result['status'] in ['error', 'warning']:
                    self.logger.warning(f"Data quality issues detected: {validation_result['status']}")
                    self.logger.info(f"Error resilience enabled - continuing processing with {validation_result.get('error_failures', 0)} errors")
                
                # Log detailed validation metrics
                if hasattr(automated_validator, 'validation_metrics') and automated_validator.validation_metrics:
                    metrics = automated_validator.validation_metrics
                    self.logger.info(
                        f"Validation metrics: {metrics.successful_rules}/{metrics.total_rules_executed} rules passed "
                        f"({metrics.overall_data_quality_score:.1f}% data quality score)"
                    )
                
                # Log validation rule execution statistics
                rule_stats = validation_result.get('rule_execution_stats', {})
                if rule_stats:
                    success_rate = (rule_stats.get('rules_executed_successfully', 0) / 
                                  max(1, rule_stats.get('total_rules_attempted', 1))) * 100
                    self.logger.info(f"Rule execution success rate: {success_rate:.1f}%")
                    
                    if rule_stats.get('execution_errors'):
                        self.logger.warning(f"Validation execution errors: {len(rule_stats['execution_errors'])}")
                        # Log error summary for troubleshooting
                        error_types = {}
                        for error in rule_stats['execution_errors']:
                            error_type = error.get('error_type', 'unknown')
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                        self.logger.info(f"Error type distribution: {error_types}")
                
                # Store validation results for reporting
                self.validation_results = validation_result
            
            # Log cleaning statistics
            cleaning_stats = self.data_cleaner.generate_cleaning_report()
            self.logger.info(f"Cleaning completed: {cleaning_stats}")
            
            # Create final checkpoint
            if self.recovery_enabled:
                checkpoint_id = create_checkpoint(
                    stage_name="Data Cleaning",
                    data_state={'data_shape': self.processed_data.shape},
                    processed_records=len(self.processed_data),
                    stage_progress=100.0,
                    metadata={'stage': 'post_cleaning', 'cleaning_stats': cleaning_stats}
                )
                self.logger.info(f"Created post-cleaning checkpoint: {checkpoint_id}")
            
            self.status.complete_stage("Data Cleaning", len(self.processed_data))
            return True
            
        except Exception as e:
            error_msg = f"Data cleaning failed: {str(e)}"
            self.logger.error(error_msg)
            self.status.add_error(error_msg)
            
            # Attempt recovery if enabled
            if self.recovery_enabled:
                failure_type = self._classify_failure_type(e)
                recovery_success = handle_failure(
                    stage_name="Data Cleaning",
                    failure_type=failure_type,
                    error_message=error_msg,
                    context={'error_type': type(e).__name__}
                )
                
                if recovery_success:
                    self.logger.info("Recovery successful, retrying cleaning stage")
                    return self._run_cleaning_stage()  # Retry
            
            return False
    
    def _run_transformation_stage(self) -> bool:
        """Run data transformation stage."""
        self.status.start_stage("Data Transformation")
        self.logger.info("Starting data transformation stage")
        
        try:
            if self.processed_data is None:
                error_msg = "No data available for transformation"
                self.logger.error(error_msg)
                self.status.add_error(error_msg)
                return False
            
            # Generate analytical tables
            self.analytical_tables = {}
            
            # Monthly sales summary
            self.analytical_tables['monthly_sales_summary'] = (
                self.transformer.create_monthly_sales_summary(self.processed_data)
            )
            
            # Top products
            self.analytical_tables['top_products'] = (
                self.transformer.create_top_products_table(self.processed_data)
            )
            
            # Regional performance
            self.analytical_tables['region_wise_performance'] = (
                self.transformer.create_region_wise_performance_table(self.processed_data)
            )
            
            # Category discount map
            self.analytical_tables['category_discount_map'] = (
                self.transformer.create_category_discount_map(self.processed_data)
            )
            
            self.logger.info(f"Generated {len(self.analytical_tables)} analytical tables")
            
            self.status.complete_stage("Data Transformation", len(self.processed_data))
            return True
            
        except Exception as e:
            error_msg = f"Data transformation failed: {str(e)}"
            self.logger.error(error_msg)
            self.status.add_error(error_msg)
            return False
    
    def _run_anomaly_detection_stage(self) -> bool:
        """Run anomaly detection stage."""
        self.status.start_stage("Anomaly Detection")
        self.logger.info("Starting anomaly detection stage")
        
        try:
            if self.processed_data is None:
                error_msg = "No data available for anomaly detection"
                self.logger.error(error_msg)
                self.status.add_error(error_msg)
                return False
            
            # Detect anomalies
            data_with_anomalies = self.anomaly_detector.detect_all_anomalies(self.processed_data)
            
            # Generate top anomaly records
            anomaly_records = self.anomaly_detector.generate_top_anomaly_records(data_with_anomalies)
            
            # Add to analytical tables
            self.analytical_tables['anomaly_records'] = anomaly_records
            
            self.logger.info(f"Detected {len(anomaly_records)} anomalous records")
            
            self.status.complete_stage("Anomaly Detection", len(anomaly_records))
            return True
            
        except Exception as e:
            error_msg = f"Anomaly detection failed: {str(e)}"
            self.logger.error(error_msg)
            self.status.add_error(error_msg)
            return False
    
    def _run_export_stage(self) -> bool:
        """Run data export stage."""
        self.status.start_stage("Data Export")
        self.logger.info("Starting data export stage")
        
        try:
            if self.processed_data is None:
                error_msg = "No data available for export"
                self.logger.error(error_msg)
                self.status.add_error(error_msg)
                return False
            
            # Export processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export main processed dataset
            main_filename = f"processed_data_{timestamp}"
            self.data_exporter.export_processed_data(self.processed_data, main_filename)
            
            # Export analytical tables
            analytical_tables_with_timestamp = {}
            for table_name, table_data in self.analytical_tables.items():
                filename = f"{table_name}_{timestamp}"
                analytical_tables_with_timestamp[filename] = table_data
            
            self.data_exporter.export_analytical_tables(analytical_tables_with_timestamp)
            
            self.logger.info("Data export completed successfully")
            
            self.status.complete_stage("Data Export", len(self.processed_data))
            return True
            
        except Exception as e:
            error_msg = f"Data export failed: {str(e)}"
            self.logger.error(error_msg)
            self.status.add_error(error_msg)
            return False
    
    def _log_pipeline_summary(self):
        """Log pipeline execution summary."""
        summary = self.status.get_summary()
        
        self.logger.info("=== Pipeline Execution Summary ===")
        self.logger.info(f"Total Duration: {summary['duration_seconds']:.2f} seconds")
        self.logger.info(f"Records Processed: {summary['total_records_processed']}")
        self.logger.info(f"Stages Completed: {len(summary['stages_completed'])}")
        self.logger.info(f"Errors: {summary['error_count']}")
        self.logger.info(f"Warnings: {summary['warning_count']}")
        
        # Log enhanced validation summary if available
        if hasattr(self, 'validation_results') and self.validation_results:
            validation_status = self.validation_results.get('status', 'unknown')
            self.logger.info(f"Data Validation Status: {validation_status}")
            
            # Log validation metrics
            if hasattr(automated_validator, 'validation_metrics') and automated_validator.validation_metrics:
                metrics = automated_validator.validation_metrics
                self.logger.info(f"Data Quality Score: {metrics.overall_data_quality_score:.1f}%")
                self.logger.info(f"Validation Rules: {metrics.successful_rules}/{metrics.total_rules_executed} passed")
            
            # Log error resilience effectiveness
            rule_stats = self.validation_results.get('rule_execution_stats', {})
            if rule_stats:
                failed_rules = rule_stats.get('rules_failed_execution', 0)
                if failed_rules > 0:
                    self.logger.info(f"Error Resilience: Pipeline continued despite {failed_rules} validation rule failures")
        
        # Log stage performance
        for stage, metrics in summary['performance_metrics'].items():
            if 'duration' in metrics:
                self.logger.info(
                    f"Stage '{stage}': {metrics['duration']:.2f}s, "
                    f"{metrics.get('records_processed', 0)} records"
                )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return self.status.get_summary()
    
    def get_analytical_tables(self) -> Dict[str, pd.DataFrame]:
        """Get generated analytical tables."""
        return self.analytical_tables.copy()
    
    def _start_monitoring_systems(self) -> None:
        """Start all monitoring systems."""
        try:
            if self.health_monitoring_enabled:
                start_health_monitoring()
                self.logger.info("Health monitoring started")
            
            if self.alerting_enabled:
                start_alerting()
                self.logger.info("Alerting system started")
            
            if self.recovery_enabled:
                pipeline_recovery_manager.start_recovery_monitoring()
                self.logger.info("Recovery monitoring started")
                
        except Exception as e:
            self.logger.error(f"Error starting monitoring systems: {e}")
    
    def _stop_monitoring_systems(self) -> None:
        """Stop all monitoring systems."""
        try:
            if self.health_monitoring_enabled:
                stop_health_monitoring()
                self.logger.info("Health monitoring stopped")
            
            if self.alerting_enabled:
                stop_alerting()
                self.logger.info("Alerting system stopped")
            
            if self.recovery_enabled:
                pipeline_recovery_manager.stop_recovery_monitoring()
                self.logger.info("Recovery monitoring stopped")
                
        except Exception as e:
            self.logger.error(f"Error stopping monitoring systems: {e}")
    
    def _attempt_pipeline_recovery(self, error: Exception, error_msg: str) -> None:
        """Attempt to recover from pipeline failure."""
        try:
            # Determine failure type based on error
            failure_type = self._classify_failure_type(error)
            
            # Attempt recovery
            recovery_success = handle_failure(
                stage_name=self.status.current_stage,
                failure_type=failure_type,
                error_message=error_msg,
                context={
                    'error_type': type(error).__name__,
                    'processed_records': self.status.total_records_processed,
                    'stage': self.status.current_stage
                }
            )
            
            if recovery_success:
                self.logger.info("Pipeline recovery attempt succeeded")
            else:
                self.logger.error("Pipeline recovery attempt failed")
                
        except Exception as e:
            self.logger.error(f"Error during recovery attempt: {e}")
    
    def _classify_failure_type(self, error: Exception) -> FailureType:
        """Classify the type of failure based on the error."""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        if 'memory' in error_msg or error_type in ['MemoryError', 'OutOfMemoryError']:
            return FailureType.MEMORY_ERROR
        elif 'timeout' in error_msg or error_type in ['TimeoutError']:
            return FailureType.TIMEOUT_ERROR
        elif 'io' in error_msg or 'file' in error_msg or error_type in ['IOError', 'FileNotFoundError']:
            return FailureType.IO_ERROR
        elif 'validation' in error_msg or 'quality' in error_msg:
            return FailureType.VALIDATION_ERROR
        elif 'corrupt' in error_msg or 'invalid' in error_msg:
            return FailureType.DATA_CORRUPTION
        else:
            return FailureType.PROCESSING_ERROR


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line interface parser."""
    parser = argparse.ArgumentParser(
        description="ETL Pipeline for E-commerce Data Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline.etl_main --input data/raw/sales_data.csv
  python -m src.pipeline.etl_main --input data/raw/sales_data.csv --config config/custom_config.yaml
  python -m src.pipeline.etl_main --input data/raw/sales_data.csv --skip-stages cleaning,export
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration file (default: config/pipeline_config.yaml)'
    )
    
    parser.add_argument(
        '--skip-stages',
        help='Comma-separated list of stages to skip (validation,ingestion,cleaning,transformation,anomaly_detection,export)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and show pipeline plan without executing'
    )
    
    parser.add_argument(
        '--status-only',
        action='store_true',
        help='Show pipeline status and exit'
    )
    
    return parser


def main():
    """Main entry point for CLI execution."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        etl_pipeline = ETLMain(config_path=args.config)
        
        # Set log level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Handle status-only request
        if args.status_only:
            status = etl_pipeline.get_status()
            print("Pipeline Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
            return
        
        # Handle dry run
        if args.dry_run:
            print("Dry run mode - validating configuration...")
            if etl_pipeline.validate_configuration():
                print("Configuration is valid")
                print(f"Input file: {args.input}")
                print(f"Skip stages: {args.skip_stages or 'None'}")
                print("Pipeline would execute successfully")
            else:
                print("Configuration validation failed")
                sys.exit(1)
            return
        
        # Parse skip stages
        skip_stages = []
        if args.skip_stages:
            skip_stages = [stage.strip() for stage in args.skip_stages.split(',')]
        
        # Run pipeline
        success = etl_pipeline.run_pipeline(args.input, skip_stages=skip_stages)
        
        if success:
            print("Pipeline completed successfully")
            sys.exit(0)
        else:
            print("Pipeline failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()