"""Robust logging and monitoring system for the data pipeline."""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path

from .logger import get_module_logger
from .config import config_manager


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityMetric:
    """Data quality metric data structure."""
    metric_name: str
    total_records: int
    valid_records: int
    invalid_records: int
    error_rate: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingStatistic:
    """Processing statistics data structure."""
    operation: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    records_processed: int
    records_per_second: Optional[float]
    memory_usage_mb: float
    status: str  # 'running', 'completed', 'failed'
    error_message: Optional[str] = None


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        """Initialize system monitor."""
        self.logger = get_module_logger("system_monitor")
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
    
    def start_monitoring(self) -> None:
        """Start system monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Log critical resource usage
                if metrics['memory_percent'] > 90:
                    self.logger.warning(f"High memory usage: {metrics['memory_percent']:.1f}%")
                
                if metrics['cpu_percent'] > 90:
                    self.logger.warning(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        return {
            'timestamp': datetime.now(),
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'memory_percent': memory.percent,
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count(),
            'disk_usage': {
                path: {
                    'total_gb': usage.total / (1024**3),
                    'used_gb': usage.used / (1024**3),
                    'free_gb': usage.free / (1024**3),
                    'percent': (usage.used / usage.total) * 100
                }
                for path in ['.', 'data'] if Path(path).exists()
                for usage in [psutil.disk_usage(path)]
            }
        }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return self._collect_system_metrics()
    
    def get_metrics_history(self, minutes: int = 10) -> List[Dict[str, Any]]:
        """Get metrics history for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            metrics for metrics in self.metrics_history
            if metrics['timestamp'] >= cutoff_time
        ]


class PipelineMonitor:
    """Comprehensive pipeline monitoring and statistics."""
    
    def __init__(self):
        """Initialize pipeline monitor."""
        self.logger = get_module_logger("pipeline_monitor")
        self.config = config_manager.config
        
        # Metrics storage
        self.performance_metrics: List[PerformanceMetric] = []
        self.data_quality_metrics: List[DataQualityMetric] = []
        self.processing_statistics: List[ProcessingStatistic] = []
        
        # Active operations tracking
        self.active_operations: Dict[str, ProcessingStatistic] = {}
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_history: List[Dict[str, Any]] = []
        
        # System monitor
        self.system_monitor = SystemMonitor()
        
        # Metrics aggregation
        self.metrics_by_category: Dict[str, List[PerformanceMetric]] = defaultdict(list)
    
    def start_monitoring(self) -> None:
        """Start all monitoring components."""
        self.system_monitor.start_monitoring()
        self.logger.info("Pipeline monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop all monitoring components."""
        self.system_monitor.stop_monitoring()
        self.logger.info("Pipeline monitoring stopped")
    
    def start_operation(self, operation_name: str, records_count: int = 0) -> str:
        """Start tracking a processing operation."""
        operation_id = f"{operation_name}_{int(time.time())}"
        
        stat = ProcessingStatistic(
            operation=operation_name,
            start_time=datetime.now(),
            end_time=None,
            duration=None,
            records_processed=records_count,
            records_per_second=None,
            memory_usage_mb=psutil.Process().memory_info().rss / (1024**2),
            status='running'
        )
        
        self.active_operations[operation_id] = stat
        self.logger.info(f"Started operation: {operation_name} (ID: {operation_id})")
        
        return operation_id
    
    def end_operation(self, operation_id: str, records_processed: int = None, 
                     error_message: str = None) -> None:
        """End tracking a processing operation."""
        if operation_id not in self.active_operations:
            self.logger.warning(f"Operation ID not found: {operation_id}")
            return
        
        stat = self.active_operations[operation_id]
        stat.end_time = datetime.now()
        stat.duration = (stat.end_time - stat.start_time).total_seconds()
        
        if records_processed is not None:
            stat.records_processed = records_processed
        
        if stat.records_processed > 0 and stat.duration > 0:
            stat.records_per_second = stat.records_processed / stat.duration
        
        if error_message:
            stat.status = 'failed'
            stat.error_message = error_message
        else:
            stat.status = 'completed'
        
        # Move to completed operations
        self.processing_statistics.append(stat)
        del self.active_operations[operation_id]
        
        # Log completion
        if stat.status == 'completed':
            self.logger.info(
                f"Completed operation: {stat.operation} "
                f"({stat.records_processed} records in {stat.duration:.2f}s, "
                f"{stat.records_per_second:.2f} records/sec)"
            )
        else:
            self.logger.error(f"Failed operation: {stat.operation} - {error_message}")
    
    def log_performance_metric(self, name: str, value: float, unit: str = "", 
                             category: str = "general", metadata: Dict[str, Any] = None) -> None:
        """Log a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            category=category,
            metadata=metadata or {}
        )
        
        self.performance_metrics.append(metric)
        self.metrics_by_category[category].append(metric)
        
        self.logger.info(f"Performance Metric - {name}: {value} {unit}")
    
    def log_data_quality_metric(self, metric_name: str, total_records: int, 
                              valid_records: int, invalid_records: int,
                              details: Dict[str, Any] = None) -> None:
        """Log data quality metrics."""
        error_rate = (invalid_records / total_records) * 100 if total_records > 0 else 0
        
        metric = DataQualityMetric(
            metric_name=metric_name,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            error_rate=error_rate,
            timestamp=datetime.now(),
            details=details or {}
        )
        
        self.data_quality_metrics.append(metric)
        
        self.logger.info(
            f"Data Quality - {metric_name}: {valid_records}/{total_records} valid "
            f"({error_rate:.2f}% error rate)"
        )
    
    def log_error(self, error_type: str, error_message: str, 
                 context: Dict[str, Any] = None) -> None:
        """Log an error occurrence."""
        self.error_counts[error_type] += 1
        
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'message': error_message,
            'context': context or {},
            'count': self.error_counts[error_type]
        }
        
        self.error_history.append(error_record)
        
        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        self.logger.error(f"Error logged - {error_type}: {error_message}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing statistics."""
        completed_ops = [s for s in self.processing_statistics if s.status == 'completed']
        failed_ops = [s for s in self.processing_statistics if s.status == 'failed']
        
        if not self.processing_statistics:
            return {'message': 'No processing operations recorded'}
        
        total_records = sum(s.records_processed for s in completed_ops)
        total_duration = sum(s.duration for s in completed_ops if s.duration)
        avg_throughput = total_records / total_duration if total_duration > 0 else 0
        
        return {
            'total_operations': len(self.processing_statistics),
            'completed_operations': len(completed_ops),
            'failed_operations': len(failed_ops),
            'active_operations': len(self.active_operations),
            'total_records_processed': total_records,
            'total_processing_time': total_duration,
            'average_throughput': avg_throughput,
            'success_rate': (len(completed_ops) / len(self.processing_statistics)) * 100,
            'recent_operations': [
                {
                    'operation': s.operation,
                    'duration': s.duration,
                    'records': s.records_processed,
                    'throughput': s.records_per_second,
                    'status': s.status
                }
                for s in sorted(self.processing_statistics, 
                              key=lambda x: x.start_time, reverse=True)[:10]
            ]
        }
    
    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get summary of data quality metrics."""
        if not self.data_quality_metrics:
            return {'message': 'No data quality metrics recorded'}
        
        recent_metrics = sorted(self.data_quality_metrics, 
                              key=lambda x: x.timestamp, reverse=True)[:10]
        
        total_records = sum(m.total_records for m in self.data_quality_metrics)
        total_valid = sum(m.valid_records for m in self.data_quality_metrics)
        overall_error_rate = ((total_records - total_valid) / total_records) * 100 if total_records > 0 else 0
        
        return {
            'total_metrics': len(self.data_quality_metrics),
            'total_records_analyzed': total_records,
            'total_valid_records': total_valid,
            'overall_error_rate': overall_error_rate,
            'recent_metrics': [
                {
                    'metric_name': m.metric_name,
                    'total_records': m.total_records,
                    'error_rate': m.error_rate,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in recent_metrics
            ]
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.performance_metrics:
            return {'message': 'No performance metrics recorded'}
        
        summary = {
            'total_metrics': len(self.performance_metrics),
            'by_category': {},
            'recent_metrics': []
        }
        
        # Metrics by category
        for category, metrics in self.metrics_by_category.items():
            if metrics:
                values = [m.value for m in metrics]
                summary['by_category'][category] = {
                    'count': len(metrics),
                    'avg_value': sum(values) / len(values),
                    'min_value': min(values),
                    'max_value': max(values)
                }
        
        # Recent metrics
        recent = sorted(self.performance_metrics, key=lambda x: x.timestamp, reverse=True)[:20]
        summary['recent_metrics'] = [
            {
                'name': m.name,
                'value': m.value,
                'unit': m.unit,
                'category': m.category,
                'timestamp': m.timestamp.isoformat()
            }
            for m in recent
        ]
        
        return summary
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors."""
        if not self.error_history:
            return {'message': 'No errors recorded'}
        
        return {
            'total_errors': len(self.error_history),
            'error_types': dict(self.error_counts),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def export_monitoring_report(self, filepath: str) -> None:
        """Export comprehensive monitoring report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'system_metrics': self.system_monitor.get_current_metrics(),
            'processing_summary': self.get_processing_summary(),
            'data_quality_summary': self.get_data_quality_summary(),
            'performance_summary': self.get_performance_summary(),
            'error_summary': self.get_error_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring report exported to {filepath}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics and statistics."""
        self.performance_metrics.clear()
        self.data_quality_metrics.clear()
        self.processing_statistics.clear()
        self.active_operations.clear()
        self.error_counts.clear()
        self.error_history.clear()
        self.metrics_by_category.clear()
        
        self.logger.info("All monitoring metrics reset")


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, monitor: PipelineMonitor, 
                 records_count: int = 0):
        """Initialize operation timer."""
        self.operation_name = operation_name
        self.monitor = monitor
        self.records_count = records_count
        self.operation_id = None
    
    def __enter__(self):
        """Start timing the operation."""
        self.operation_id = self.monitor.start_operation(
            self.operation_name, self.records_count
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing the operation."""
        error_message = None
        if exc_type is not None:
            error_message = str(exc_val)
        
        self.monitor.end_operation(
            self.operation_id, 
            self.records_count, 
            error_message
        )
    
    def update_records_count(self, count: int) -> None:
        """Update the records count during operation."""
        self.records_count = count


# Global monitoring instance
pipeline_monitor = PipelineMonitor()


# Convenience functions
def start_monitoring() -> None:
    """Start pipeline monitoring."""
    pipeline_monitor.start_monitoring()


def stop_monitoring() -> None:
    """Stop pipeline monitoring."""
    pipeline_monitor.stop_monitoring()


def log_performance_metric(name: str, value: float, unit: str = "", 
                         category: str = "general", metadata: Dict[str, Any] = None) -> None:
    """Log a performance metric."""
    pipeline_monitor.log_performance_metric(name, value, unit, category, metadata)


def log_data_quality_metric(metric_name: str, total_records: int, 
                          valid_records: int, invalid_records: int,
                          details: Dict[str, Any] = None) -> None:
    """Log data quality metrics."""
    pipeline_monitor.log_data_quality_metric(
        metric_name, total_records, valid_records, invalid_records, details
    )


def log_error(error_type: str, error_message: str, context: Dict[str, Any] = None) -> None:
    """Log an error occurrence."""
    pipeline_monitor.log_error(error_type, error_message, context)


def time_operation(operation_name: str, records_count: int = 0) -> OperationTimer:
    """Create an operation timer context manager."""
    return OperationTimer(operation_name, pipeline_monitor, records_count)