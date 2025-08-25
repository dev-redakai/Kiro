"""Logging infrastructure with different log levels."""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Any, List, Dict
from .config import config_manager


class PipelineLogger:
    """Custom logger for the data pipeline."""
    
    def __init__(self, name: str = "pipeline", config_path: Optional[str] = None):
        """Initialize the pipeline logger."""
        self.name = name
        self.logger = logging.getLogger(name)
        self.config = config_manager.config
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up the logger with appropriate handlers and formatters."""
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(self.config.log_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / self.config.log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, *args, **kwargs)
    
    def log_processing_start(self, operation: str, details: str = "") -> None:
        """Log the start of a processing operation."""
        self.info(f"Starting {operation}" + (f": {details}" if details else ""))
    
    def log_processing_end(self, operation: str, details: str = "") -> None:
        """Log the end of a processing operation."""
        self.info(f"Completed {operation}" + (f": {details}" if details else ""))
    
    def log_data_quality_issue(self, issue_type: str, details: str, severity: str = "WARNING") -> None:
        """Log data quality issues."""
        log_method = getattr(self, severity.lower(), self.warning)
        log_method(f"Data Quality Issue - {issue_type}: {details}")
    
    def log_memory_usage(self, operation: str, memory_mb: float) -> None:
        """Log memory usage information."""
        self.info(f"Memory usage for {operation}: {memory_mb:.2f} MB")
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log performance metrics."""
        self.info(f"Performance Metric - {metric_name}: {value:.2f} {unit}")
    
    def log_anomaly_detection(self, anomaly_count: int, total_records: int) -> None:
        """Log anomaly detection results."""
        percentage = (anomaly_count / total_records) * 100 if total_records > 0 else 0
        self.info(f"Anomaly Detection: Found {anomaly_count} anomalies out of {total_records} records ({percentage:.2f}%)")
    
    def log_processing_statistics(self, operation: str, records_processed: int, 
                                duration: float, memory_usage: float) -> None:
        """Log detailed processing statistics."""
        throughput = records_processed / duration if duration > 0 else 0
        self.info(f"Processing Stats - {operation}: {records_processed} records, "
                 f"{duration:.2f}s, {throughput:.2f} records/sec, {memory_usage:.2f}MB memory")
    
    def log_data_validation_results(self, field_name: str, total_count: int, 
                                  valid_count: int, invalid_count: int, 
                                  corrections_applied: int = 0) -> None:
        """Log data validation results for a specific field."""
        error_rate = (invalid_count / total_count) * 100 if total_count > 0 else 0
        self.info(f"Validation - {field_name}: {valid_count}/{total_count} valid "
                 f"({error_rate:.2f}% error rate), {corrections_applied} corrections applied")
    
    def log_chunk_processing(self, chunk_number: int, chunk_size: int, 
                           total_chunks: int, processing_time: float) -> None:
        """Log chunk processing progress."""
        progress = (chunk_number / total_chunks) * 100 if total_chunks > 0 else 0
        self.info(f"Chunk {chunk_number}/{total_chunks} ({progress:.1f}%): "
                 f"{chunk_size} records processed in {processing_time:.2f}s")
    
    def log_transformation_metrics(self, transformation_name: str, 
                                 input_records: int, output_records: int,
                                 processing_time: float) -> None:
        """Log transformation metrics."""
        records_per_sec = input_records / processing_time if processing_time > 0 else 0
        self.info(f"Transformation - {transformation_name}: {input_records} -> {output_records} records "
                 f"in {processing_time:.2f}s ({records_per_sec:.2f} records/sec)")
    
    def log_error_recovery(self, error_type: str, recovery_action: str, success: bool) -> None:
        """Log error recovery attempts."""
        status = "successful" if success else "failed"
        self.info(f"Error Recovery - {error_type}: {recovery_action} ({status})")
    
    def log_system_resource_usage(self, cpu_percent: float, memory_percent: float, 
                                disk_usage_percent: float) -> None:
        """Log system resource usage."""
        self.debug(f"System Resources - CPU: {cpu_percent:.1f}%, "
                  f"Memory: {memory_percent:.1f}%, Disk: {disk_usage_percent:.1f}%")
    
    def log_pipeline_stage(self, stage_name: str, status: str, details: str = "") -> None:
        """Log pipeline stage transitions."""
        self.info(f"Pipeline Stage - {stage_name}: {status.upper()}" + 
                 (f" - {details}" if details else ""))
    
    def log_data_export(self, format_type: str, file_path: str, record_count: int, 
                       file_size_mb: float) -> None:
        """Log data export operations."""
        self.info(f"Data Export - {format_type}: {record_count} records exported to {file_path} "
                 f"({file_size_mb:.2f}MB)")
    
    def log_configuration_change(self, parameter: str, old_value: Any, new_value: Any) -> None:
        """Log configuration changes."""
        self.info(f"Configuration Change - {parameter}: {old_value} -> {new_value}")
    
    def log_dashboard_access(self, user_info: str, page: str, load_time: float) -> None:
        """Log dashboard access and performance."""
        self.info(f"Dashboard Access - {user_info} accessed {page} (loaded in {load_time:.2f}s)")
    
    def log_cache_operation(self, operation: str, cache_key: str, hit: bool = None) -> None:
        """Log cache operations."""
        if hit is not None:
            status = "HIT" if hit else "MISS"
            self.debug(f"Cache {operation} - {cache_key}: {status}")
        else:
            self.debug(f"Cache {operation} - {cache_key}")
    
    def log_batch_summary(self, batch_id: str, total_records: int, successful_records: int,
                         failed_records: int, processing_time: float) -> None:
        """Log batch processing summary."""
        success_rate = (successful_records / total_records) * 100 if total_records > 0 else 0
        self.info(f"Batch Summary - {batch_id}: {successful_records}/{total_records} successful "
                 f"({success_rate:.2f}%), {failed_records} failed, {processing_time:.2f}s total")
    
    def log_validation_context(self, rule_name: str, field_name: str, data_type: str, 
                             sample_values: List[Any], execution_time: float,
                             problematic_values: List[Any] = None) -> None:
        """Log detailed validation context for debugging."""
        problematic_values = problematic_values or []
        self.debug(
            f"Validation Context - {rule_name}:\n"
            f"  Field: {field_name} (type: {data_type})\n"
            f"  Execution Time: {execution_time*1000:.2f}ms\n"
            f"  Sample Values: {sample_values[:3]}\n"
            f"  Problematic Values: {problematic_values[:3]}"
        )
    
    def log_validation_error_with_context(self, rule_name: str, field_name: str, 
                                        error_type: str, error_message: str,
                                        problematic_values: List[Any],
                                        suggested_actions: List[str],
                                        execution_time: float = 0.0) -> None:
        """Log validation error with comprehensive context and actionable information."""
        self.error(
            f"VALIDATION ERROR - {rule_name}:\n"
            f"  Field: {field_name}\n"
            f"  Error: {error_type} - {error_message}\n"
            f"  Execution Time: {execution_time*1000:.2f}ms\n"
            f"  Problematic Values: {problematic_values[:5]}\n"
            f"  Suggested Actions: {', '.join(suggested_actions[:3])}"
        )
    
    def log_validation_metrics_summary(self, total_rules: int, successful_rules: int,
                                     failed_rules: int, total_records: int,
                                     valid_records: int, invalid_records: int,
                                     execution_time: float, memory_usage_mb: float,
                                     data_quality_score: float) -> None:
        """Log comprehensive validation metrics summary."""
        self.info(
            f"VALIDATION METRICS SUMMARY:\n"
            f"  Rules: {successful_rules}/{total_rules} successful ({failed_rules} failed)\n"
            f"  Records: {valid_records:,}/{total_records:,} valid ({invalid_records:,} invalid)\n"
            f"  Quality Score: {data_quality_score:.2f}%\n"
            f"  Performance: {execution_time:.2f}s execution, {memory_usage_mb:.2f}MB memory"
        )
    
    def log_problematic_values_analysis(self, field_name: str, total_invalid: int,
                                      error_patterns: Dict[str, int],
                                      sample_values: List[Any],
                                      recommendations: List[str]) -> None:
        """Log detailed analysis of problematic values for a field."""
        self.warning(
            f"PROBLEMATIC VALUES ANALYSIS - {field_name}:\n"
            f"  Total Invalid Records: {total_invalid:,}\n"
            f"  Error Patterns: {dict(sorted(error_patterns.items(), key=lambda x: x[1], reverse=True))}\n"
            f"  Sample Values: {sample_values[:5]}\n"
            f"  Recommendations: {recommendations[:3]}"
        )
    
    def log_actionable_insights(self, insights: List[str]) -> None:
        """Log actionable insights from validation analysis."""
        if insights:
            self.info("ACTIONABLE INSIGHTS:")
            for i, insight in enumerate(insights, 1):
                self.info(f"  {i}. {insight}")
        else:
            self.info("VALIDATION HEALTH: All metrics are within acceptable ranges.")
    
    def log_structured_error_report(self, report_path: str, total_errors: int,
                                  most_problematic_field: str, 
                                  recommendations_count: int) -> None:
        """Log structured error report generation."""
        self.info(
            f"Structured Error Report Generated:\n"
            f"  Report Path: {report_path}\n"
            f"  Total Errors Analyzed: {total_errors}\n"
            f"  Most Problematic Field: {most_problematic_field}\n"
            f"  Recommendations Generated: {recommendations_count}"
        )


class LoggerFactory:
    """Factory for creating loggers with consistent configuration."""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str) -> PipelineLogger:
        """Get or create a logger with the given name."""
        if name not in cls._loggers:
            cls._loggers[name] = PipelineLogger(name)
        return cls._loggers[name]
    
    @classmethod
    def get_module_logger(cls, module_name: str) -> PipelineLogger:
        """Get a logger for a specific module."""
        return cls.get_logger(f"pipeline.{module_name}")
    
    @classmethod
    def reset_loggers(cls) -> None:
        """Reset all loggers (useful for testing)."""
        cls._loggers.clear()


# Convenience functions for getting loggers
def get_logger(name: str = "pipeline") -> PipelineLogger:
    """Get the main pipeline logger."""
    return LoggerFactory.get_logger(name)


def get_module_logger(module_name: str) -> PipelineLogger:
    """Get a logger for a specific module."""
    return LoggerFactory.get_module_logger(module_name)


# Create logs directory
Path("logs").mkdir(exist_ok=True)