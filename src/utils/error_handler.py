"""Comprehensive error handling system for the data pipeline."""

import traceback
import psutil
import gc
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd

from .logger import get_module_logger
from .config import config_manager


class ErrorCategory(Enum):
    """Categories of errors that can occur in the pipeline."""
    DATA_QUALITY = "data_quality"
    MEMORY = "memory"
    PROCESSING = "processing"
    DASHBOARD = "dashboard"
    IO = "io"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    error_type: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None
    recovery_action: Optional[str] = None
    resolved: bool = False


class DataQualityError(Exception):
    """Exception for data quality issues."""
    def __init__(self, message: str, field: str = None, value: Any = None, correction: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.correction = correction


class PipelineMemoryError(Exception):
    """Exception for memory-related issues."""
    def __init__(self, message: str, current_usage: float = None, max_usage: float = None):
        super().__init__(message)
        self.current_usage = current_usage
        self.max_usage = max_usage


class ProcessingError(Exception):
    """Exception for processing-related issues."""
    def __init__(self, message: str, operation: str = None, data_info: Dict[str, Any] = None):
        super().__init__(message)
        self.operation = operation
        self.data_info = data_info or {}


class DashboardError(Exception):
    """Exception for dashboard-related issues."""
    def __init__(self, message: str, component: str = None, fallback_available: bool = False):
        super().__init__(message)
        self.component = component
        self.fallback_available = fallback_available


class ErrorHandler:
    """Comprehensive error handler for various error categories."""
    
    def __init__(self):
        """Initialize the error handler."""
        self.logger = get_module_logger("error_handler")
        self.config = config_manager.config
        self.error_history: List[ErrorRecord] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.DATA_QUALITY: [
                self._apply_data_correction,
                self._use_default_value,
                self._skip_invalid_record
            ],
            ErrorCategory.MEMORY: [
                self._reduce_chunk_size,
                self._force_garbage_collection,
                self._clear_cache,
                self._use_memory_mapping
            ],
            ErrorCategory.PROCESSING: [
                self._retry_operation,
                self._skip_problematic_chunk,
                self._use_fallback_method
            ],
            ErrorCategory.DASHBOARD: [
                self._use_cached_data,
                self._show_error_message,
                self._use_fallback_visualization
            ]
        }
    
    def handle_error(self, error: Exception, category: ErrorCategory, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Dict[str, Any] = None) -> Any:
        """Handle an error with appropriate recovery strategy."""
        context = context or {}
        
        # Create error record
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            error_type=type(error).__name__,
            message=str(error),
            details=context,
            traceback=traceback.format_exc()
        )
        
        # Log the error
        self._log_error(error_record)
        
        # Store error record
        self.error_history.append(error_record)
        
        # Apply recovery strategy
        recovery_result = self._apply_recovery_strategy(error, category, context)
        
        if recovery_result is not None:
            error_record.recovery_action = recovery_result.get('action', 'Unknown')
            error_record.resolved = recovery_result.get('resolved', False)
            
            if error_record.resolved:
                self.logger.info(f"Error resolved using strategy: {error_record.recovery_action}")
            
            return recovery_result.get('result')
        
        # If no recovery strategy worked, re-raise for critical errors
        if severity == ErrorSeverity.CRITICAL and recovery_result is None:
            raise error
        
        return None
    
    def handle_data_quality_error(self, error: DataQualityError, 
                                context: Dict[str, Any] = None) -> Any:
        """Handle data quality errors with correction strategies."""
        context = context or {}
        context.update({
            'field': error.field,
            'value': error.value,
            'correction': error.correction
        })
        
        return self.handle_error(error, ErrorCategory.DATA_QUALITY, 
                               ErrorSeverity.LOW, context)
    
    def handle_memory_error(self, error: PipelineMemoryError, 
                          context: Dict[str, Any] = None) -> Any:
        """Handle memory errors with automatic recovery."""
        context = context or {}
        context.update({
            'current_usage': error.current_usage,
            'max_usage': error.max_usage,
            'available_memory': psutil.virtual_memory().available / (1024**3)  # GB
        })
        
        return self.handle_error(error, ErrorCategory.MEMORY, 
                               ErrorSeverity.HIGH, context)
    
    def handle_processing_error(self, error: ProcessingError, 
                              context: Dict[str, Any] = None) -> Any:
        """Handle processing errors with graceful continuation."""
        context = context or {}
        context.update({
            'operation': error.operation,
            'data_info': error.data_info
        })
        
        return self.handle_error(error, ErrorCategory.PROCESSING, 
                               ErrorSeverity.MEDIUM, context)
    
    def handle_dashboard_error(self, error: DashboardError, 
                             context: Dict[str, Any] = None) -> Any:
        """Handle dashboard errors with fallback mechanisms."""
        context = context or {}
        context.update({
            'component': error.component,
            'fallback_available': error.fallback_available
        })
        
        return self.handle_error(error, ErrorCategory.DASHBOARD, 
                               ErrorSeverity.MEDIUM, context)
    
    def _log_error(self, error_record: ErrorRecord) -> None:
        """Log error record with appropriate level."""
        log_message = (f"[{error_record.category.value.upper()}] "
                      f"{error_record.error_type}: {error_record.message}")
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log additional details
        if error_record.details:
            self.logger.debug(f"Error details: {error_record.details}")
        
        if error_record.traceback:
            self.logger.debug(f"Traceback: {error_record.traceback}")
    
    def _apply_recovery_strategy(self, error: Exception, category: ErrorCategory, 
                               context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply recovery strategies for the given error category."""
        strategies = self.recovery_strategies.get(category, [])
        
        for strategy in strategies:
            try:
                result = strategy(error, context)
                if result is not None:
                    return {
                        'action': strategy.__name__,
                        'resolved': True,
                        'result': result
                    }
            except Exception as recovery_error:
                self.logger.debug(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
                continue
        
        return None
    
    # Data Quality Recovery Strategies
    def _apply_data_correction(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Apply data correction if available."""
        if isinstance(error, DataQualityError) and error.correction is not None:
            self.logger.info(f"Applying data correction for field {error.field}: "
                           f"{error.value} -> {error.correction}")
            return error.correction
        return None
    
    def _skip_invalid_record(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Skip invalid record and continue processing."""
        self.logger.info("Skipping invalid record and continuing processing")
        return "SKIP_RECORD"
    
    def _use_default_value(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Use default value for invalid data."""
        if isinstance(error, DataQualityError):
            field = error.field
            default_values = {
                'quantity': 0,
                'unit_price': 0.0,
                'discount_percent': 0.0,
                'product_name': 'Unknown Product',
                'category': 'Unknown',
                'region': 'Unknown',
                'customer_email': None
            }
            default_value = default_values.get(field)
            if default_value is not None:
                self.logger.info(f"Using default value for {field}: {default_value}")
                return default_value
        return None
    
    # Memory Recovery Strategies
    def _reduce_chunk_size(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Reduce chunk size to manage memory usage."""
        current_chunk_size = context.get('chunk_size', self.config.chunk_size)
        new_chunk_size = max(1000, current_chunk_size // 2)
        
        # Only proceed if we can actually reduce the chunk size
        if new_chunk_size >= current_chunk_size:
            return None
        
        self.logger.info(f"Reducing chunk size from {current_chunk_size} to {new_chunk_size}")
        
        # Update config temporarily
        self.config.chunk_size = new_chunk_size
        
        return {'new_chunk_size': new_chunk_size}
    
    def _force_garbage_collection(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Force garbage collection to free memory."""
        self.logger.info("Forcing garbage collection to free memory")
        collected = gc.collect()
        self.logger.info(f"Garbage collection freed {collected} objects")
        return {'objects_collected': collected}
    
    def _clear_cache(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Clear any cached data to free memory."""
        self.logger.info("Clearing caches to free memory")
        # This would clear any application-specific caches
        return {'cache_cleared': True}
    
    def _use_memory_mapping(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Use memory mapping for large files."""
        self.logger.info("Switching to memory-mapped file access")
        return {'use_memory_mapping': True}
    
    # Processing Recovery Strategies
    def _retry_operation(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Retry the failed operation."""
        retry_count = context.get('retry_count', 0)
        max_retries = 3
        
        if retry_count < max_retries:
            self.logger.info(f"Retrying operation (attempt {retry_count + 1}/{max_retries})")
            return {'retry': True, 'retry_count': retry_count + 1}
        
        return None
    
    def _skip_problematic_chunk(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Skip the problematic data chunk."""
        self.logger.warning("Skipping problematic data chunk")
        return {'skip_chunk': True}
    
    def _use_fallback_method(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Use alternative processing method."""
        self.logger.info("Using fallback processing method")
        return {'use_fallback': True}
    
    # Dashboard Recovery Strategies
    def _use_cached_data(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Use cached data when fresh data is unavailable."""
        self.logger.info("Using cached data due to loading error")
        return {'use_cache': True}
    
    def _show_error_message(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Show error message to user."""
        error_message = f"Error in {context.get('component', 'dashboard')}: {str(error)}"
        return {'error_message': error_message}
    
    def _use_fallback_visualization(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Use simpler visualization when complex one fails."""
        self.logger.info("Using fallback visualization")
        return {'use_fallback_viz': True}
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        if not self.error_history:
            return {'total_errors': 0}
        
        summary = {
            'total_errors': len(self.error_history),
            'by_category': {},
            'by_severity': {},
            'resolved_count': sum(1 for e in self.error_history if e.resolved),
            'recent_errors': []
        }
        
        # Count by category
        for category in ErrorCategory:
            count = sum(1 for e in self.error_history if e.category == category)
            if count > 0:
                summary['by_category'][category.value] = count
        
        # Count by severity
        for severity in ErrorSeverity:
            count = sum(1 for e in self.error_history if e.severity == severity)
            if count > 0:
                summary['by_severity'][severity.value] = count
        
        # Recent errors (last 10)
        recent = sorted(self.error_history, key=lambda x: x.timestamp, reverse=True)[:10]
        summary['recent_errors'] = [
            {
                'timestamp': e.timestamp.isoformat(),
                'category': e.category.value,
                'severity': e.severity.value,
                'type': e.error_type,
                'message': e.message,
                'resolved': e.resolved
            }
            for e in recent
        ]
        
        return summary
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        self.error_history.clear()
        self.logger.info("Error history cleared")
    
    def export_error_log(self, filepath: str) -> None:
        """Export error history to a file."""
        import json
        
        error_data = []
        for error in self.error_history:
            error_data.append({
                'timestamp': error.timestamp.isoformat(),
                'category': error.category.value,
                'severity': error.severity.value,
                'error_type': error.error_type,
                'message': error.message,
                'details': error.details,
                'recovery_action': error.recovery_action,
                'resolved': error.resolved
            })
        
        with open(filepath, 'w') as f:
            json.dump(error_data, f, indent=2)
        
        self.logger.info(f"Error log exported to {filepath}")


# Global error handler instance
error_handler = ErrorHandler()


# Convenience functions
def handle_data_quality_error(error: DataQualityError, context: Dict[str, Any] = None) -> Any:
    """Handle data quality error using global error handler."""
    return error_handler.handle_data_quality_error(error, context)


def handle_memory_error(error: PipelineMemoryError, context: Dict[str, Any] = None) -> Any:
    """Handle memory error using global error handler."""
    return error_handler.handle_memory_error(error, context)


def handle_processing_error(error: ProcessingError, context: Dict[str, Any] = None) -> Any:
    """Handle processing error using global error handler."""
    return error_handler.handle_processing_error(error, context)


def handle_dashboard_error(error: DashboardError, context: Dict[str, Any] = None) -> Any:
    """Handle dashboard error using global error handler."""
    return error_handler.handle_dashboard_error(error, context)