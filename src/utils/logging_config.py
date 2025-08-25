"""Advanced logging configuration with monitoring integration."""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from .config import config_manager


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class MetricsHandler(logging.Handler):
    """Custom handler that sends metrics to monitoring system."""
    
    def __init__(self):
        """Initialize metrics handler."""
        super().__init__()
        self.metrics_count = {
            'DEBUG': 0,
            'INFO': 0,
            'WARNING': 0,
            'ERROR': 0,
            'CRITICAL': 0
        }
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record and update metrics."""
        try:
            # Count log levels
            self.metrics_count[record.levelname] += 1
            
            # Send error metrics to monitoring system
            if record.levelname in ['ERROR', 'CRITICAL']:
                # Import here to avoid circular imports
                from .monitoring import pipeline_monitor
                pipeline_monitor.log_error(
                    error_type=f"{record.module}.{record.funcName}",
                    error_message=record.getMessage(),
                    context={
                        'level': record.levelname,
                        'module': record.module,
                        'function': record.funcName,
                        'line': record.lineno
                    }
                )
        except Exception:
            # Don't let logging errors break the application
            pass
    
    def get_metrics(self) -> Dict[str, int]:
        """Get current metrics count."""
        return self.metrics_count.copy()


class PerformanceFilter(logging.Filter):
    """Filter to add performance context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance context to log record."""
        try:
            import psutil
            process = psutil.Process()
            
            # Add performance context
            record.memory_mb = process.memory_info().rss / (1024**2)
            record.cpu_percent = process.cpu_percent()
            
        except Exception:
            # Don't fail if we can't get performance info
            record.memory_mb = 0
            record.cpu_percent = 0
        
        return True


class DataQualityFilter(logging.Filter):
    """Filter to identify and categorize data quality log messages."""
    
    DATA_QUALITY_KEYWORDS = [
        'validation', 'cleaning', 'standardization', 'anomaly',
        'duplicate', 'invalid', 'missing', 'null', 'error_rate'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add data quality context to relevant log records."""
        message_lower = record.getMessage().lower()
        
        # Check if this is a data quality related message
        is_data_quality = any(keyword in message_lower for keyword in self.DATA_QUALITY_KEYWORDS)
        
        if is_data_quality:
            record.category = 'data_quality'
            
            # Try to extract metrics from the message
            if 'error rate' in message_lower or 'error_rate' in message_lower:
                record.subcategory = 'error_rate'
            elif 'validation' in message_lower:
                record.subcategory = 'validation'
            elif 'cleaning' in message_lower:
                record.subcategory = 'cleaning'
            elif 'anomaly' in message_lower:
                record.subcategory = 'anomaly'
            else:
                record.subcategory = 'general'
        
        return True


class ProcessingFilter(logging.Filter):
    """Filter to identify and categorize processing log messages."""
    
    PROCESSING_KEYWORDS = [
        'processing', 'transformation', 'chunk', 'batch',
        'records/sec', 'throughput', 'completed', 'started'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add processing context to relevant log records."""
        message_lower = record.getMessage().lower()
        
        # Check if this is a processing related message
        is_processing = any(keyword in message_lower for keyword in self.PROCESSING_KEYWORDS)
        
        if is_processing:
            record.category = 'processing'
            
            # Try to extract processing stage
            if 'ingestion' in message_lower:
                record.stage = 'ingestion'
            elif 'cleaning' in message_lower:
                record.stage = 'cleaning'
            elif 'transformation' in message_lower:
                record.stage = 'transformation'
            elif 'export' in message_lower:
                record.stage = 'export'
            else:
                record.stage = 'general'
        
        return True


class AdvancedLoggingConfig:
    """Advanced logging configuration with monitoring integration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize advanced logging configuration."""
        self.config = config_manager.config
        self.metrics_handler = MetricsHandler()
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Set up advanced logging configuration."""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Set root log level
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        root_logger.setLevel(log_level)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        json_formatter = JSONFormatter()
        
        # Console handler with simple format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        console_handler.addFilter(PerformanceFilter())
        root_logger.addHandler(console_handler)
        
        # Main log file with detailed format
        main_file_handler = logging.handlers.RotatingFileHandler(
            log_dir / self.config.log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        main_file_handler.setLevel(log_level)
        main_file_handler.setFormatter(detailed_formatter)
        main_file_handler.addFilter(PerformanceFilter())
        root_logger.addHandler(main_file_handler)
        
        # Error log file (errors and critical only)
        error_file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_file_handler)
        
        # Data quality log file
        data_quality_handler = logging.handlers.RotatingFileHandler(
            log_dir / "data_quality.log",
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=5
        )
        data_quality_handler.setLevel(logging.INFO)
        data_quality_handler.setFormatter(detailed_formatter)
        data_quality_handler.addFilter(DataQualityFilter())
        
        # Only log data quality messages to this handler
        class DataQualityOnlyFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'category') and record.category == 'data_quality'
        
        data_quality_handler.addFilter(DataQualityOnlyFilter())
        root_logger.addHandler(data_quality_handler)
        
        # Processing log file
        processing_handler = logging.handlers.RotatingFileHandler(
            log_dir / "processing.log",
            maxBytes=30 * 1024 * 1024,  # 30MB
            backupCount=5
        )
        processing_handler.setLevel(logging.INFO)
        processing_handler.setFormatter(detailed_formatter)
        processing_handler.addFilter(ProcessingFilter())
        
        # Only log processing messages to this handler
        class ProcessingOnlyFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'category') and record.category == 'processing'
        
        processing_handler.addFilter(ProcessingOnlyFilter())
        root_logger.addHandler(processing_handler)
        
        # JSON log file for structured logging
        json_handler = logging.handlers.RotatingFileHandler(
            log_dir / "pipeline.json",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=3
        )
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(json_formatter)
        json_handler.addFilter(PerformanceFilter())
        json_handler.addFilter(DataQualityFilter())
        json_handler.addFilter(ProcessingFilter())
        root_logger.addHandler(json_handler)
        
        # Metrics handler
        self.metrics_handler.setLevel(logging.WARNING)
        root_logger.addHandler(self.metrics_handler)
        
        # Prevent propagation to avoid duplicate logs
        root_logger.propagate = False
        
        logging.info("Advanced logging configuration initialized")
    
    def get_logging_metrics(self) -> Dict[str, Any]:
        """Get logging metrics."""
        return {
            'log_counts_by_level': self.metrics_handler.get_metrics(),
            'log_files': self._get_log_file_info()
        }
    
    def _get_log_file_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about log files."""
        log_dir = Path("logs")
        log_files = {}
        
        for log_file in log_dir.glob("*.log"):
            try:
                stat = log_file.stat()
                log_files[log_file.name] = {
                    'size_mb': stat.st_size / (1024**2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'lines': self._count_lines(log_file)
                }
            except Exception as e:
                log_files[log_file.name] = {'error': str(e)}
        
        return log_files
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file efficiently."""
        try:
            with open(file_path, 'rb') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    def rotate_logs(self) -> None:
        """Manually rotate all log files."""
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.doRollover()
        
        logging.info("Log files rotated manually")
    
    def set_log_level(self, level: str) -> None:
        """Change log level dynamically."""
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Update all handlers
        for handler in logging.getLogger().handlers:
            handler.setLevel(log_level)
        
        logging.getLogger().setLevel(log_level)
        
        # Update config
        self.config.log_level = level.upper()
        
        logging.info(f"Log level changed to {level.upper()}")


# Global logging configuration instance
advanced_logging = AdvancedLoggingConfig()


# Convenience functions
def get_logging_metrics() -> Dict[str, Any]:
    """Get current logging metrics."""
    return advanced_logging.get_logging_metrics()


def rotate_logs() -> None:
    """Rotate all log files."""
    advanced_logging.rotate_logs()


def set_log_level(level: str) -> None:
    """Set log level dynamically."""
    advanced_logging.set_log_level(level)