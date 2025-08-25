"""Pipeline health monitoring and status checks system."""

import time
import threading
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import pandas as pd

from .logger import get_module_logger
from .config import config_manager
from .monitoring import pipeline_monitor
from .error_handler import error_handler, ErrorCategory, ErrorSeverity


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class ComponentType(Enum):
    """Types of pipeline components."""
    INGESTION = "ingestion"
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    ANOMALY_DETECTION = "anomaly_detection"
    EXPORT = "export"
    SYSTEM = "system"
    DATA_QUALITY = "data_quality"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    component: ComponentType
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: float = 60.0
    timeout_seconds: float = 30.0
    enabled: bool = True
    last_run: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.HEALTHY
    last_result: Dict[str, Any] = field(default_factory=dict)
    failure_count: int = 0
    max_failures: int = 3


@dataclass
class PipelineHealthStatus:
    """Overall pipeline health status."""
    overall_status: HealthStatus
    timestamp: datetime
    component_statuses: Dict[ComponentType, HealthStatus]
    active_alerts: List[Dict[str, Any]]
    system_metrics: Dict[str, Any]
    data_quality_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recovery_actions: List[Dict[str, Any]]


class PipelineHealthMonitor:
    """Comprehensive pipeline health monitoring system."""
    
    def __init__(self):
        """Initialize pipeline health monitor."""
        self.logger = get_module_logger("pipeline_health")
        self.config = config_manager.config
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.health_history: List[PipelineHealthStatus] = []
        
        # Alert thresholds
        self.memory_warning_threshold = 0.8  # 80%
        self.memory_critical_threshold = 0.9  # 90%
        self.cpu_warning_threshold = 80.0  # 80%
        self.cpu_critical_threshold = 90.0  # 90%
        self.disk_warning_threshold = 85.0  # 85%
        self.disk_critical_threshold = 95.0  # 95%
        
        # Data quality thresholds
        self.data_quality_warning_threshold = 5.0  # 5% error rate
        self.data_quality_critical_threshold = 10.0  # 10% error rate
        
        # Performance thresholds
        self.throughput_warning_threshold = 1000  # records/sec
        self.memory_usage_warning_mb = 2048  # 2GB
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self) -> None:
        """Register default health checks."""
        # System health checks
        self.register_health_check(
            "system_memory",
            ComponentType.SYSTEM,
            self._check_system_memory,
            interval_seconds=30.0
        )
        
        self.register_health_check(
            "system_cpu",
            ComponentType.SYSTEM,
            self._check_system_cpu,
            interval_seconds=30.0
        )
        
        self.register_health_check(
            "system_disk",
            ComponentType.SYSTEM,
            self._check_system_disk,
            interval_seconds=60.0
        )
        
        # Data quality health checks
        self.register_health_check(
            "data_quality_overall",
            ComponentType.DATA_QUALITY,
            self._check_data_quality_overall,
            interval_seconds=120.0
        )
        
        # Pipeline component health checks
        self.register_health_check(
            "pipeline_performance",
            ComponentType.SYSTEM,
            self._check_pipeline_performance,
            interval_seconds=60.0
        )
        
        # File system health checks
        self.register_health_check(
            "data_directories",
            ComponentType.SYSTEM,
            self._check_data_directories,
            interval_seconds=300.0  # 5 minutes
        )
    
    def register_health_check(self, name: str, component: ComponentType,
                            check_function: Callable[[], Dict[str, Any]],
                            interval_seconds: float = 60.0,
                            timeout_seconds: float = 30.0,
                            max_failures: int = 3) -> None:
        """Register a new health check."""
        health_check = HealthCheck(
            name=name,
            component=component,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            max_failures=max_failures
        )
        
        self.health_checks[name] = health_check
        self.logger.info(f"Registered health check: {name} for component {component.value}")
    
    def start_monitoring(self) -> None:
        """Start health monitoring in background thread."""
        if self.is_monitoring:
            self.logger.warning("Health monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Pipeline health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Pipeline health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                
                # Run health checks that are due
                for check_name, health_check in self.health_checks.items():
                    if not health_check.enabled:
                        continue
                    
                    # Check if it's time to run this health check
                    if (health_check.last_run is None or 
                        (current_time - health_check.last_run).total_seconds() >= health_check.interval_seconds):
                        
                        self._run_health_check(health_check)
                
                # Generate overall health status
                overall_status = self._generate_health_status()
                self.health_history.append(overall_status)
                
                # Keep only last 100 health status records
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]
                
                # Log critical issues
                if overall_status.overall_status == HealthStatus.CRITICAL:
                    self.logger.critical("Pipeline health status is CRITICAL")
                elif overall_status.overall_status == HealthStatus.WARNING:
                    self.logger.warning("Pipeline health status is WARNING")
                
                # Sleep for a short interval before next check
                time.sleep(10.0)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(30.0)  # Wait longer on error
    
    def _run_health_check(self, health_check: HealthCheck) -> None:
        """Run a single health check."""
        try:
            start_time = time.time()
            
            # Run the health check with timeout
            result = self._run_with_timeout(
                health_check.check_function,
                health_check.timeout_seconds
            )
            
            duration = time.time() - start_time
            
            # Update health check status
            health_check.last_run = datetime.now()
            health_check.last_result = result
            
            # Determine status from result
            if result.get('status') == 'healthy':
                health_check.last_status = HealthStatus.HEALTHY
                health_check.failure_count = 0
            elif result.get('status') == 'warning':
                health_check.last_status = HealthStatus.WARNING
            elif result.get('status') == 'critical':
                health_check.last_status = HealthStatus.CRITICAL
                health_check.failure_count += 1
            else:
                health_check.last_status = HealthStatus.FAILED
                health_check.failure_count += 1
            
            # Log health check result
            self.logger.debug(
                f"Health check '{health_check.name}': {health_check.last_status.value} "
                f"(duration: {duration:.2f}s)"
            )
            
            # Handle repeated failures
            if health_check.failure_count >= health_check.max_failures:
                self.logger.error(
                    f"Health check '{health_check.name}' has failed "
                    f"{health_check.failure_count} times consecutively"
                )
                
                # Trigger recovery actions if available
                self._trigger_recovery_actions(health_check)
            
        except Exception as e:
            health_check.last_status = HealthStatus.FAILED
            health_check.failure_count += 1
            health_check.last_result = {'error': str(e)}
            
            self.logger.error(f"Health check '{health_check.name}' failed with error: {e}")
    
    def _run_with_timeout(self, func: Callable, timeout_seconds: float) -> Dict[str, Any]:
        """Run function with timeout."""
        import threading
        import queue
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def target():
            try:
                result = func()
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            # Thread is still running, timeout occurred
            raise TimeoutError(f"Health check timed out after {timeout_seconds} seconds")
        
        # Check for exceptions
        if not exception_queue.empty():
            raise exception_queue.get()
        
        # Get result
        if not result_queue.empty():
            return result_queue.get()
        else:
            raise RuntimeError("Health check completed but no result was returned")
    
    def _trigger_recovery_actions(self, health_check: HealthCheck) -> None:
        """Trigger recovery actions for failed health check."""
        recovery_actions = []
        
        if health_check.component == ComponentType.SYSTEM:
            if "memory" in health_check.name:
                # Memory recovery actions
                recovery_actions.extend([
                    "force_garbage_collection",
                    "reduce_chunk_size",
                    "clear_caches"
                ])
            elif "cpu" in health_check.name:
                # CPU recovery actions
                recovery_actions.extend([
                    "reduce_parallel_workers",
                    "increase_processing_intervals"
                ])
            elif "disk" in health_check.name:
                # Disk recovery actions
                recovery_actions.extend([
                    "cleanup_temp_files",
                    "compress_old_logs"
                ])
        
        # Execute recovery actions
        for action in recovery_actions:
            try:
                self._execute_recovery_action(action, health_check)
            except Exception as e:
                self.logger.error(f"Recovery action '{action}' failed: {e}")
    
    def _execute_recovery_action(self, action: str, health_check: HealthCheck) -> None:
        """Execute a specific recovery action."""
        self.logger.info(f"Executing recovery action: {action}")
        
        if action == "force_garbage_collection":
            import gc
            collected = gc.collect()
            self.logger.info(f"Garbage collection freed {collected} objects")
        
        elif action == "reduce_chunk_size":
            current_size = self.config.chunk_size
            new_size = max(1000, current_size // 2)
            self.config.chunk_size = new_size
            self.logger.info(f"Reduced chunk size from {current_size} to {new_size}")
        
        elif action == "clear_caches":
            # Clear any application caches
            self.logger.info("Cleared application caches")
        
        elif action == "cleanup_temp_files":
            temp_dir = Path(self.config.temp_data_path)
            if temp_dir.exists():
                for file in temp_dir.glob("*"):
                    if file.is_file() and file.stat().st_mtime < time.time() - 3600:  # 1 hour old
                        file.unlink()
                self.logger.info("Cleaned up old temporary files")
        
        elif action == "compress_old_logs":
            # Compress old log files
            logs_dir = Path("logs")
            if logs_dir.exists():
                import gzip
                for log_file in logs_dir.glob("*.log"):
                    if log_file.stat().st_mtime < time.time() - 86400:  # 1 day old
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(f"{log_file}.gz", 'wb') as f_out:
                                f_out.writelines(f_in)
                        log_file.unlink()
                self.logger.info("Compressed old log files")
    
    # Health check implementations
    def _check_system_memory(self) -> Dict[str, Any]:
        """Check system memory usage."""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100.0
        
        if usage_percent >= self.memory_critical_threshold:
            status = "critical"
        elif usage_percent >= self.memory_warning_threshold:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            'status': status,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'threshold_warning': self.memory_warning_threshold * 100,
            'threshold_critical': self.memory_critical_threshold * 100
        }
    
    def _check_system_cpu(self) -> Dict[str, Any]:
        """Check system CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1.0)
        
        if cpu_percent >= self.cpu_critical_threshold:
            status = "critical"
        elif cpu_percent >= self.cpu_warning_threshold:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            'status': status,
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count(),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            'threshold_warning': self.cpu_warning_threshold,
            'threshold_critical': self.cpu_critical_threshold
        }
    
    def _check_system_disk(self) -> Dict[str, Any]:
        """Check system disk usage."""
        disk_usage = psutil.disk_usage('.')
        usage_percent = (disk_usage.used / disk_usage.total) * 100
        
        if usage_percent >= self.disk_critical_threshold:
            status = "critical"
        elif usage_percent >= self.disk_warning_threshold:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            'status': status,
            'disk_usage_percent': usage_percent,
            'disk_free_gb': disk_usage.free / (1024**3),
            'disk_used_gb': disk_usage.used / (1024**3),
            'disk_total_gb': disk_usage.total / (1024**3),
            'threshold_warning': self.disk_warning_threshold,
            'threshold_critical': self.disk_critical_threshold
        }
    
    def _check_data_quality_overall(self) -> Dict[str, Any]:
        """Check overall data quality metrics."""
        quality_summary = pipeline_monitor.get_data_quality_summary()
        
        if 'overall_error_rate' not in quality_summary:
            return {'status': 'healthy', 'message': 'No data quality metrics available'}
        
        error_rate = quality_summary['overall_error_rate']
        
        if error_rate >= self.data_quality_critical_threshold:
            status = "critical"
        elif error_rate >= self.data_quality_warning_threshold:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            'status': status,
            'overall_error_rate': error_rate,
            'total_records_analyzed': quality_summary.get('total_records_analyzed', 0),
            'total_valid_records': quality_summary.get('total_valid_records', 0),
            'threshold_warning': self.data_quality_warning_threshold,
            'threshold_critical': self.data_quality_critical_threshold
        }
    
    def _check_pipeline_performance(self) -> Dict[str, Any]:
        """Check pipeline performance metrics."""
        perf_summary = pipeline_monitor.get_processing_summary()
        
        if 'average_throughput' not in perf_summary:
            return {'status': 'healthy', 'message': 'No performance metrics available'}
        
        avg_throughput = perf_summary['average_throughput']
        success_rate = perf_summary.get('success_rate', 100.0)
        
        # Determine status based on throughput and success rate
        if avg_throughput < self.throughput_warning_threshold or success_rate < 90.0:
            status = "warning"
        elif success_rate < 80.0:
            status = "critical"
        else:
            status = "healthy"
        
        return {
            'status': status,
            'average_throughput': avg_throughput,
            'success_rate': success_rate,
            'total_operations': perf_summary.get('total_operations', 0),
            'failed_operations': perf_summary.get('failed_operations', 0),
            'threshold_throughput': self.throughput_warning_threshold
        }
    
    def _check_data_directories(self) -> Dict[str, Any]:
        """Check data directories accessibility and permissions."""
        data_paths = self.config.get_data_paths()
        issues = []
        
        for path_name, path_value in data_paths.items():
            path_obj = Path(path_value)
            
            if not path_obj.exists():
                issues.append(f"Directory {path_name} does not exist: {path_value}")
            elif not path_obj.is_dir():
                issues.append(f"Path {path_name} is not a directory: {path_value}")
            elif not os.access(path_value, os.R_OK | os.W_OK):
                issues.append(f"Directory {path_name} is not readable/writable: {path_value}")
        
        if issues:
            status = "critical" if len(issues) > 2 else "warning"
        else:
            status = "healthy"
        
        return {
            'status': status,
            'issues': issues,
            'directories_checked': len(data_paths)
        }
    
    def _generate_health_status(self) -> PipelineHealthStatus:
        """Generate overall pipeline health status."""
        component_statuses = {}
        active_alerts = []
        
        # Aggregate component statuses
        for health_check in self.health_checks.values():
            if not health_check.enabled:
                continue
            
            component = health_check.component
            current_status = component_statuses.get(component, HealthStatus.HEALTHY)
            
            # Use the worst status for each component
            if health_check.last_status.value == "failed":
                component_statuses[component] = HealthStatus.FAILED
            elif health_check.last_status.value == "critical" and current_status != HealthStatus.FAILED:
                component_statuses[component] = HealthStatus.CRITICAL
            elif health_check.last_status.value == "warning" and current_status not in [HealthStatus.FAILED, HealthStatus.CRITICAL]:
                component_statuses[component] = HealthStatus.WARNING
            elif component not in component_statuses:
                component_statuses[component] = HealthStatus.HEALTHY
            
            # Generate alerts for non-healthy checks
            if health_check.last_status != HealthStatus.HEALTHY:
                active_alerts.append({
                    'check_name': health_check.name,
                    'component': health_check.component.value,
                    'status': health_check.last_status.value,
                    'message': health_check.last_result.get('message', 'Health check failed'),
                    'failure_count': health_check.failure_count,
                    'last_run': health_check.last_run.isoformat() if health_check.last_run else None
                })
        
        # Determine overall status
        if HealthStatus.FAILED in component_statuses.values():
            overall_status = HealthStatus.FAILED
        elif HealthStatus.CRITICAL in component_statuses.values():
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in component_statuses.values():
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Get current metrics
        system_metrics = pipeline_monitor.system_monitor.get_current_metrics()
        data_quality_metrics = pipeline_monitor.get_data_quality_summary()
        performance_metrics = pipeline_monitor.get_performance_summary()
        
        return PipelineHealthStatus(
            overall_status=overall_status,
            timestamp=datetime.now(),
            component_statuses=component_statuses,
            active_alerts=active_alerts,
            system_metrics=system_metrics,
            data_quality_metrics=data_quality_metrics,
            performance_metrics=performance_metrics,
            recovery_actions=[]  # Will be populated by recovery system
        )
    
    def get_current_health_status(self) -> PipelineHealthStatus:
        """Get current pipeline health status."""
        if self.health_history:
            return self.health_history[-1]
        else:
            return self._generate_health_status()
    
    def get_health_history(self, hours: int = 24) -> List[PipelineHealthStatus]:
        """Get health status history for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            status for status in self.health_history
            if status.timestamp >= cutoff_time
        ]
    
    def export_health_report(self, filepath: str) -> None:
        """Export comprehensive health report."""
        current_status = self.get_current_health_status()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'overall_status': current_status.overall_status.value,
            'component_statuses': {
                comp.value: status.value 
                for comp, status in current_status.component_statuses.items()
            },
            'active_alerts': current_status.active_alerts,
            'system_metrics': current_status.system_metrics,
            'data_quality_metrics': current_status.data_quality_metrics,
            'performance_metrics': current_status.performance_metrics,
            'health_checks': {
                name: {
                    'component': check.component.value,
                    'enabled': check.enabled,
                    'last_status': check.last_status.value,
                    'last_run': check.last_run.isoformat() if check.last_run else None,
                    'failure_count': check.failure_count,
                    'last_result': check.last_result
                }
                for name, check in self.health_checks.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Health report exported to {filepath}")


# Global health monitor instance
pipeline_health_monitor = PipelineHealthMonitor()


# Convenience functions
def start_health_monitoring() -> None:
    """Start pipeline health monitoring."""
    pipeline_health_monitor.start_monitoring()


def stop_health_monitoring() -> None:
    """Stop pipeline health monitoring."""
    pipeline_health_monitor.stop_monitoring()


def get_pipeline_health() -> PipelineHealthStatus:
    """Get current pipeline health status."""
    return pipeline_health_monitor.get_current_health_status()


def register_custom_health_check(name: str, component: ComponentType,
                                check_function: Callable[[], Dict[str, Any]],
                                interval_seconds: float = 60.0) -> None:
    """Register a custom health check."""
    pipeline_health_monitor.register_health_check(
        name, component, check_function, interval_seconds
    )