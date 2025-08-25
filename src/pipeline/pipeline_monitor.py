"""
Pipeline monitoring and health check system.

This module provides comprehensive monitoring capabilities for the ETL pipeline
including health checks, performance monitoring, data quality validation,
and recovery mechanisms.
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import psutil

from ..utils.logger import get_module_logger
from ..utils.config import ConfigManager
from ..utils.error_handler import ErrorHandler


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    total_records_processed: int = 0
    processing_rate_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    active_threads: int = 0
    error_count: int = 0
    warning_count: int = 0
    stage_durations: Dict[str, float] = None
    
    def __post_init__(self):
        if self.stage_durations is None:
            self.stage_durations = {}


class PipelineHealthChecker:
    """
    Performs health checks on pipeline components and system resources.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the health checker.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        self.logger = get_module_logger("pipeline_health_checker")
        
        # Health check thresholds
        self.memory_warning_threshold = 80.0  # 80%
        self.memory_critical_threshold = 95.0  # 95%
        self.cpu_warning_threshold = 80.0  # 80%
        self.cpu_critical_threshold = 95.0  # 95%
        self.disk_warning_threshold = 85.0  # 85%
        self.disk_critical_threshold = 95.0  # 95%
        
        # Health check registry
        self.health_checks = {
            'system_memory': self._check_system_memory,
            'system_cpu': self._check_system_cpu,
            'disk_space': self._check_disk_space,
            'data_directories': self._check_data_directories,
            'configuration': self._check_configuration,
            'dependencies': self._check_dependencies
        }
    
    def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary mapping check name to result
        """
        self.logger.info("Running all health checks")
        results = {}
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[check_name] = result
                self.logger.debug(f"Health check '{check_name}': {result.status.value}")
            except Exception as e:
                error_result = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now(),
                    details={'error': str(e)}
                )
                results[check_name] = error_result
                self.logger.error(f"Health check '{check_name}' failed: {e}")
        
        return results
    
    def _check_system_memory(self) -> HealthCheckResult:
        """Check system memory usage."""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        
        if usage_percent >= self.memory_critical_threshold:
            status = HealthStatus.CRITICAL
            message = f"Critical memory usage: {usage_percent:.1f}%"
        elif usage_percent >= self.memory_warning_threshold:
            status = HealthStatus.WARNING
            message = f"High memory usage: {usage_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {usage_percent:.1f}%"
        
        return HealthCheckResult(
            name="system_memory",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details={
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'usage_percent': usage_percent
            }
        )
    
    def _check_system_cpu(self) -> HealthCheckResult:
        """Check system CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent >= self.cpu_critical_threshold:
            status = HealthStatus.CRITICAL
            message = f"Critical CPU usage: {cpu_percent:.1f}%"
        elif cpu_percent >= self.cpu_warning_threshold:
            status = HealthStatus.WARNING
            message = f"High CPU usage: {cpu_percent:.1f}%"
        else: