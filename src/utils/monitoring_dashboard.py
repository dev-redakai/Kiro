"""Monitoring dashboard integration for pipeline health and performance."""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from .logger import get_module_logger
from .monitoring import pipeline_monitor
from .pipeline_health import pipeline_health_monitor, get_pipeline_health
from .pipeline_recovery import pipeline_recovery_manager
from .pipeline_alerting import pipeline_alerting_system, get_active_alerts
from .config import config_manager


class MonitoringDashboardProvider:
    """Provides data for monitoring dashboard."""
    
    def __init__(self):
        """Initialize monitoring dashboard provider."""
        self.logger = get_module_logger("monitoring_dashboard")
        self.config = config_manager.config
        
        # Dashboard data cache
        self.cache_duration = 30.0  # 30 seconds
        self.last_update = 0
        self.cached_data = {}
    
    def get_dashboard_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        current_time = time.time()
        
        # Check if cache is still valid
        if not force_refresh and (current_time - self.last_update) < self.cache_duration:
            return self.cached_data
        
        try:
            # Collect all monitoring data
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': self._get_system_status(),
                'pipeline_health': self._get_pipeline_health_data(),
                'performance_metrics': self._get_performance_metrics(),
                'data_quality_metrics': self._get_data_quality_metrics(),
                'active_alerts': self._get_active_alerts_data(),
                'recovery_status': self._get_recovery_status(),
                'processing_summary': self._get_processing_summary(),
                'resource_usage': self._get_resource_usage(),
                'recent_activities': self._get_recent_activities()
            }
            
            # Update cache
            self.cached_data = dashboard_data
            self.last_update = current_time
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error collecting dashboard data: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        try:
            health_status = get_pipeline_health()
            
            return {
                'overall_status': health_status.overall_status.value,
                'status_color': self._get_status_color(health_status.overall_status.value),
                'component_count': len(health_status.component_statuses),
                'healthy_components': sum(
                    1 for status in health_status.component_statuses.values()
                    if status.value == 'healthy'
                ),
                'warning_components': sum(
                    1 for status in health_status.component_statuses.values()
                    if status.value == 'warning'
                ),
                'critical_components': sum(
                    1 for status in health_status.component_statuses.values()
                    if status.value in ['critical', 'failed']
                ),
                'last_updated': health_status.timestamp.isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def _get_pipeline_health_data(self) -> Dict[str, Any]:
        """Get pipeline health data."""
        try:
            health_status = get_pipeline_health()
            
            # Get health history for trend analysis
            health_history = pipeline_health_monitor.get_health_history(hours=24)
            
            # Calculate health trend
            health_trend = []
            for status in health_history[-20:]:  # Last 20 data points
                health_trend.append({
                    'timestamp': status.timestamp.isoformat(),
                    'status': status.overall_status.value,
                    'alert_count': len(status.active_alerts)
                })
            
            return {
                'current_status': health_status.overall_status.value,
                'component_statuses': {
                    comp.value: {
                        'status': status.value,
                        'color': self._get_status_color(status.value)
                    }
                    for comp, status in health_status.component_statuses.items()
                },
                'active_alerts_count': len(health_status.active_alerts),
                'health_trend': health_trend,
                'system_metrics': health_status.system_metrics,
                'last_check': health_status.timestamp.isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting pipeline health data: {e}")
            return {'error': str(e)}
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            perf_summary = pipeline_monitor.get_performance_summary()
            processing_summary = pipeline_monitor.get_processing_summary()
            
            # Calculate performance trends
            recent_operations = processing_summary.get('recent_operations', [])
            throughput_trend = []
            
            for op in recent_operations[-10:]:  # Last 10 operations
                if op.get('throughput'):
                    throughput_trend.append({
                        'operation': op['operation'],
                        'throughput': op['throughput'],
                        'duration': op.get('duration', 0)
                    })
            
            return {
                'total_operations': processing_summary.get('total_operations', 0),
                'average_throughput': processing_summary.get('average_throughput', 0),
                'success_rate': processing_summary.get('success_rate', 100),
                'total_records_processed': processing_summary.get('total_records_processed', 0),
                'total_processing_time': processing_summary.get('total_processing_time', 0),
                'throughput_trend': throughput_trend,
                'performance_by_category': perf_summary.get('by_category', {}),
                'recent_metrics': perf_summary.get('recent_metrics', [])[:10]
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {'error': str(e)}
    
    def _get_data_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics."""
        try:
            quality_summary = pipeline_monitor.get_data_quality_summary()
            
            if 'message' in quality_summary:
                return {'message': quality_summary['message']}
            
            # Calculate quality trend
            recent_metrics = quality_summary.get('recent_metrics', [])
            quality_trend = []
            
            for metric in recent_metrics[-10:]:
                quality_trend.append({
                    'timestamp': metric['timestamp'],
                    'metric_name': metric['metric_name'],
                    'error_rate': metric['error_rate']
                })
            
            return {
                'total_records_analyzed': quality_summary.get('total_records_analyzed', 0),
                'overall_error_rate': quality_summary.get('overall_error_rate', 0),
                'total_valid_records': quality_summary.get('total_valid_records', 0),
                'quality_status': self._get_quality_status(quality_summary.get('overall_error_rate', 0)),
                'quality_trend': quality_trend,
                'recent_metrics': recent_metrics[:5]
            }
        except Exception as e:
            self.logger.error(f"Error getting data quality metrics: {e}")
            return {'error': str(e)}
    
    def _get_active_alerts_data(self) -> Dict[str, Any]:
        """Get active alerts data."""
        try:
            active_alerts = get_active_alerts()
            alert_summary = pipeline_alerting_system.get_alert_summary(hours=24)
            
            # Group alerts by severity
            alerts_by_severity = {'critical': [], 'error': [], 'warning': [], 'info': []}
            
            for alert in active_alerts:
                severity = alert.severity.value
                if severity in alerts_by_severity:
                    alerts_by_severity[severity].append({
                        'alert_id': alert.alert_id,
                        'title': alert.title,
                        'timestamp': alert.timestamp.isoformat(),
                        'acknowledged': alert.acknowledged
                    })
            
            return {
                'total_active_alerts': len(active_alerts),
                'alerts_by_severity': alerts_by_severity,
                'alert_summary_24h': alert_summary,
                'recent_alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'title': alert.title,
                        'severity': alert.severity.value,
                        'timestamp': alert.timestamp.isoformat(),
                        'resolved': alert.resolved,
                        'acknowledged': alert.acknowledged
                    }
                    for alert in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:10]
                ]
            }
        except Exception as e:
            self.logger.error(f"Error getting active alerts data: {e}")
            return {'error': str(e)}
    
    def _get_recovery_status(self) -> Dict[str, Any]:
        """Get recovery system status."""
        try:
            recovery_summary = pipeline_recovery_manager.get_recovery_summary()
            
            return {
                'total_checkpoints': recovery_summary.get('total_checkpoints', 0),
                'recent_failures_24h': recovery_summary.get('recent_failures_24h', 0),
                'recovery_success_rate': recovery_summary.get('recovery_success_rate', 100),
                'monitoring_active': recovery_summary.get('monitoring_active', False),
                'recent_failures': recovery_summary.get('recent_failures', [])[:5]
            }
        except Exception as e:
            self.logger.error(f"Error getting recovery status: {e}")
            return {'error': str(e)}
    
    def _get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        try:
            processing_summary = pipeline_monitor.get_processing_summary()
            
            if 'message' in processing_summary:
                return {'message': processing_summary['message']}
            
            return {
                'total_operations': processing_summary.get('total_operations', 0),
                'completed_operations': processing_summary.get('completed_operations', 0),
                'failed_operations': processing_summary.get('failed_operations', 0),
                'active_operations': processing_summary.get('active_operations', 0),
                'success_rate': processing_summary.get('success_rate', 100),
                'average_throughput': processing_summary.get('average_throughput', 0),
                'total_processing_time': processing_summary.get('total_processing_time', 0),
                'recent_operations': processing_summary.get('recent_operations', [])[:5]
            }
        except Exception as e:
            self.logger.error(f"Error getting processing summary: {e}")
            return {'error': str(e)}
    
    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get system resource usage."""
        try:
            system_metrics = pipeline_monitor.system_monitor.get_current_metrics()
            
            return {
                'memory': {
                    'total_gb': system_metrics.get('memory_total_gb', 0),
                    'used_gb': system_metrics.get('memory_used_gb', 0),
                    'available_gb': system_metrics.get('memory_available_gb', 0),
                    'percent': system_metrics.get('memory_percent', 0),
                    'status': self._get_resource_status(system_metrics.get('memory_percent', 0), 'memory')
                },
                'cpu': {
                    'percent': system_metrics.get('cpu_percent', 0),
                    'count': system_metrics.get('cpu_count', 0),
                    'status': self._get_resource_status(system_metrics.get('cpu_percent', 0), 'cpu')
                },
                'disk': system_metrics.get('disk_usage', {}),
                'timestamp': system_metrics.get('timestamp', datetime.now()).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting resource usage: {e}")
            return {'error': str(e)}
    
    def _get_recent_activities(self) -> List[Dict[str, Any]]:
        """Get recent pipeline activities."""
        try:
            activities = []
            
            # Get recent processing operations
            processing_summary = pipeline_monitor.get_processing_summary()
            recent_ops = processing_summary.get('recent_operations', [])
            
            for op in recent_ops[:5]:
                activities.append({
                    'type': 'processing',
                    'description': f"Completed {op['operation']}",
                    'status': op['status'],
                    'details': f"{op.get('records', 0)} records in {op.get('duration', 0):.2f}s",
                    'timestamp': datetime.now().isoformat()  # Would need actual timestamp from operation
                })
            
            # Get recent alerts
            active_alerts = get_active_alerts()
            for alert in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:3]:
                activities.append({
                    'type': 'alert',
                    'description': f"Alert: {alert.title}",
                    'status': alert.severity.value,
                    'details': f"Severity: {alert.severity.value}",
                    'timestamp': alert.timestamp.isoformat()
                })
            
            # Sort by timestamp (most recent first)
            activities.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return activities[:10]
            
        except Exception as e:
            self.logger.error(f"Error getting recent activities: {e}")
            return []
    
    def _get_status_color(self, status: str) -> str:
        """Get color code for status."""
        color_map = {
            'healthy': '#28a745',    # Green
            'warning': '#ffc107',    # Yellow
            'critical': '#dc3545',   # Red
            'failed': '#6c757d',     # Gray
            'error': '#dc3545'       # Red
        }
        return color_map.get(status, '#6c757d')
    
    def _get_quality_status(self, error_rate: float) -> str:
        """Get quality status based on error rate."""
        if error_rate <= 1.0:
            return 'excellent'
        elif error_rate <= 5.0:
            return 'good'
        elif error_rate <= 10.0:
            return 'fair'
        else:
            return 'poor'
    
    def _get_resource_status(self, usage_percent: float, resource_type: str) -> str:
        """Get resource status based on usage percentage."""
        if resource_type == 'memory':
            if usage_percent <= 70:
                return 'normal'
            elif usage_percent <= 85:
                return 'warning'
            else:
                return 'critical'
        elif resource_type == 'cpu':
            if usage_percent <= 80:
                return 'normal'
            elif usage_percent <= 90:
                return 'warning'
            else:
                return 'critical'
        else:
            return 'normal'
    
    def export_dashboard_data(self, filepath: str) -> None:
        """Export dashboard data to file."""
        try:
            dashboard_data = self.get_dashboard_data(force_refresh=True)
            
            with open(filepath, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            self.logger.info(f"Dashboard data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting dashboard data: {e}")
    
    def get_health_check_status(self) -> Dict[str, Any]:
        """Get health check endpoint status for API."""
        try:
            health_status = get_pipeline_health()
            
            return {
                'status': health_status.overall_status.value,
                'timestamp': health_status.timestamp.isoformat(),
                'checks': {
                    comp.value: status.value
                    for comp, status in health_status.component_statuses.items()
                },
                'alerts': len(health_status.active_alerts)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global dashboard provider instance
monitoring_dashboard = MonitoringDashboardProvider()


# Convenience functions
def get_dashboard_data(force_refresh: bool = False) -> Dict[str, Any]:
    """Get dashboard data."""
    return monitoring_dashboard.get_dashboard_data(force_refresh)


def get_health_status() -> Dict[str, Any]:
    """Get health status for API endpoint."""
    return monitoring_dashboard.get_health_check_status()


def export_monitoring_report(filepath: str) -> None:
    """Export comprehensive monitoring report."""
    monitoring_dashboard.export_dashboard_data(filepath)