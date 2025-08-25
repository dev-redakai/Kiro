"""Pipeline performance monitoring and alerting system."""

import time
import threading
import smtplib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .logger import get_module_logger
from .config import config_manager
from .monitoring import pipeline_monitor
from .pipeline_health import pipeline_health_monitor, HealthStatus


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    EMAIL = "email"
    FILE = "file"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"


@dataclass
class AlertRule:
    """Alert rule definition."""
    rule_id: str
    name: str
    description: str
    condition_function: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 15
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    message: str
    details: Dict[str, Any]
    channels_sent: List[AlertChannel] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class PipelineAlertingSystem:
    """Comprehensive pipeline alerting system."""
    
    def __init__(self):
        """Initialize pipeline alerting system."""
        self.logger = get_module_logger("pipeline_alerting")
        self.config = config_manager.config
        
        # Alert rules and history
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Alerting state
        self.is_monitoring = False
        self.monitor_thread = None
        self.check_interval = 60.0  # Check every minute
        
        # Alert channels configuration
        self.email_config = {
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'username': '',
            'password': '',
            'from_email': 'pipeline@company.com',
            'to_emails': []
        }
        
        # Alert storage
        self.alerts_dir = Path("logs/alerts")
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # Register default alert rules
        self._register_default_alert_rules()
    
    def _register_default_alert_rules(self) -> None:
        """Register default alert rules."""
        
        # High memory usage alert
        self.register_alert_rule(
            "high_memory_usage",
            "High Memory Usage",
            "System memory usage is above 85%",
            self._check_high_memory_usage,
            AlertSeverity.WARNING,
            [AlertChannel.LOG, AlertChannel.FILE],
            cooldown_minutes=10
        )
        
        # Critical memory usage alert
        self.register_alert_rule(
            "critical_memory_usage",
            "Critical Memory Usage",
            "System memory usage is above 95%",
            self._check_critical_memory_usage,
            AlertSeverity.CRITICAL,
            [AlertChannel.LOG, AlertChannel.FILE, AlertChannel.EMAIL],
            cooldown_minutes=5
        )
        
        # High CPU usage alert
        self.register_alert_rule(
            "high_cpu_usage",
            "High CPU Usage",
            "CPU usage is above 90% for extended period",
            self._check_high_cpu_usage,
            AlertSeverity.WARNING,
            [AlertChannel.LOG, AlertChannel.FILE],
            cooldown_minutes=15
        )
        
        # Low throughput alert
        self.register_alert_rule(
            "low_throughput",
            "Low Processing Throughput",
            "Processing throughput is below expected levels",
            self._check_low_throughput,
            AlertSeverity.WARNING,
            [AlertChannel.LOG, AlertChannel.FILE],
            cooldown_minutes=20
        )
        
        # High error rate alert
        self.register_alert_rule(
            "high_error_rate",
            "High Error Rate",
            "Pipeline error rate is above acceptable threshold",
            self._check_high_error_rate,
            AlertSeverity.ERROR,
            [AlertChannel.LOG, AlertChannel.FILE, AlertChannel.EMAIL],
            cooldown_minutes=10
        )
        
        # Data quality degradation alert
        self.register_alert_rule(
            "data_quality_degradation",
            "Data Quality Degradation",
            "Data quality metrics show significant degradation",
            self._check_data_quality_degradation,
            AlertSeverity.ERROR,
            [AlertChannel.LOG, AlertChannel.FILE],
            cooldown_minutes=30
        )
        
        # Pipeline failure alert
        self.register_alert_rule(
            "pipeline_failure",
            "Pipeline Failure",
            "Pipeline has failed or stopped unexpectedly",
            self._check_pipeline_failure,
            AlertSeverity.CRITICAL,
            [AlertChannel.LOG, AlertChannel.FILE, AlertChannel.EMAIL],
            cooldown_minutes=5
        )
        
        # Disk space alert
        self.register_alert_rule(
            "low_disk_space",
            "Low Disk Space",
            "Available disk space is below 10%",
            self._check_low_disk_space,
            AlertSeverity.WARNING,
            [AlertChannel.LOG, AlertChannel.FILE],
            cooldown_minutes=60
        )
    
    def register_alert_rule(self, rule_id: str, name: str, description: str,
                          condition_function: Callable[[Dict[str, Any]], bool],
                          severity: AlertSeverity, channels: List[AlertChannel],
                          cooldown_minutes: int = 15, metadata: Dict[str, Any] = None) -> None:
        """Register a new alert rule."""
        rule = AlertRule(
            rule_id=rule_id,
            name=name,
            description=description,
            condition_function=condition_function,
            severity=severity,
            channels=channels,
            cooldown_minutes=cooldown_minutes,
            metadata=metadata or {}
        )
        
        self.alert_rules[rule_id] = rule
        self.logger.info(f"Registered alert rule: {rule_id} ({severity.value})")
    
    def start_monitoring(self) -> None:
        """Start alert monitoring in background thread."""
        if self.is_monitoring:
            self.logger.warning("Alert monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Pipeline alerting system started")
    
    def stop_monitoring(self) -> None:
        """Stop alert monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Pipeline alerting system stopped")
    
    def _monitoring_loop(self) -> None:
        """Main alert monitoring loop."""
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                
                # Collect current metrics
                metrics = self._collect_metrics()
                
                # Check each alert rule
                for rule_id, rule in self.alert_rules.items():
                    if not rule.enabled:
                        continue
                    
                    # Check cooldown period
                    if (rule.last_triggered and 
                        (current_time - rule.last_triggered).total_seconds() < rule.cooldown_minutes * 60):
                        continue
                    
                    try:
                        # Evaluate rule condition
                        if rule.condition_function(metrics):
                            self._trigger_alert(rule, metrics)
                    
                    except Exception as e:
                        self.logger.error(f"Error evaluating alert rule {rule_id}: {e}")
                
                # Check for alert resolution
                self._check_alert_resolution(metrics)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in alert monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current metrics for alert evaluation."""
        metrics = {}
        
        try:
            # System metrics
            system_metrics = pipeline_monitor.system_monitor.get_current_metrics()
            metrics['system'] = system_metrics
            
            # Processing metrics
            processing_summary = pipeline_monitor.get_processing_summary()
            metrics['processing'] = processing_summary
            
            # Data quality metrics
            quality_summary = pipeline_monitor.get_data_quality_summary()
            metrics['data_quality'] = quality_summary
            
            # Performance metrics
            performance_summary = pipeline_monitor.get_performance_summary()
            metrics['performance'] = performance_summary
            
            # Health status
            health_status = pipeline_health_monitor.get_current_health_status()
            metrics['health'] = {
                'overall_status': health_status.overall_status.value,
                'component_statuses': {
                    comp.value: status.value 
                    for comp, status in health_status.component_statuses.items()
                },
                'active_alerts_count': len(health_status.active_alerts)
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics for alerting: {e}")
        
        return metrics
    
    def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]) -> None:
        """Trigger an alert."""
        alert_id = f"{rule.rule_id}_{int(time.time())}"
        
        # Generate alert message
        alert_message = self._generate_alert_message(rule, metrics)
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            timestamp=datetime.now(),
            severity=rule.severity,
            title=rule.name,
            message=alert_message,
            details=metrics
        )
        
        # Update rule state
        rule.last_triggered = datetime.now()
        rule.trigger_count += 1
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send alert through configured channels
        for channel in rule.channels:
            try:
                self._send_alert(alert, channel)
                alert.channels_sent.append(channel)
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel.value}: {e}")
        
        self.logger.warning(f"Alert triggered: {rule.name} (ID: {alert_id})")
    
    def _generate_alert_message(self, rule: AlertRule, metrics: Dict[str, Any]) -> str:
        """Generate alert message based on rule and metrics."""
        message_parts = [
            f"Alert: {rule.name}",
            f"Severity: {rule.severity.value.upper()}",
            f"Description: {rule.description}",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Add relevant metrics based on rule type
        if "memory" in rule.rule_id.lower():
            system_metrics = metrics.get('system', {})
            if 'memory_percent' in system_metrics:
                message_parts.append(f"Current Memory Usage: {system_metrics['memory_percent']:.1f}%")
                message_parts.append(f"Available Memory: {system_metrics.get('memory_available_gb', 0):.2f} GB")
        
        elif "cpu" in rule.rule_id.lower():
            system_metrics = metrics.get('system', {})
            if 'cpu_percent' in system_metrics:
                message_parts.append(f"Current CPU Usage: {system_metrics['cpu_percent']:.1f}%")
        
        elif "throughput" in rule.rule_id.lower():
            processing_metrics = metrics.get('processing', {})
            if 'average_throughput' in processing_metrics:
                message_parts.append(f"Current Throughput: {processing_metrics['average_throughput']:.2f} records/sec")
        
        elif "error" in rule.rule_id.lower():
            processing_metrics = metrics.get('processing', {})
            if 'failed_operations' in processing_metrics:
                message_parts.append(f"Failed Operations: {processing_metrics['failed_operations']}")
                message_parts.append(f"Success Rate: {processing_metrics.get('success_rate', 0):.1f}%")
        
        elif "quality" in rule.rule_id.lower():
            quality_metrics = metrics.get('data_quality', {})
            if 'overall_error_rate' in quality_metrics:
                message_parts.append(f"Data Quality Error Rate: {quality_metrics['overall_error_rate']:.2f}%")
        
        return "\n".join(message_parts)
    
    def _send_alert(self, alert: Alert, channel: AlertChannel) -> None:
        """Send alert through specified channel."""
        if channel == AlertChannel.LOG:
            self._send_log_alert(alert)
        elif channel == AlertChannel.FILE:
            self._send_file_alert(alert)
        elif channel == AlertChannel.EMAIL:
            self._send_email_alert(alert)
        elif channel == AlertChannel.WEBHOOK:
            self._send_webhook_alert(alert)
        elif channel == AlertChannel.DASHBOARD:
            self._send_dashboard_alert(alert)
    
    def _send_log_alert(self, alert: Alert) -> None:
        """Send alert to log."""
        log_method = getattr(self.logger, alert.severity.value, self.logger.info)
        log_method(f"ALERT: {alert.title} - {alert.message}")
    
    def _send_file_alert(self, alert: Alert) -> None:
        """Send alert to file."""
        alert_file = self.alerts_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
        alert_data = {
            'alert_id': alert.alert_id,
            'rule_id': alert.rule_id,
            'timestamp': alert.timestamp.isoformat(),
            'severity': alert.severity.value,
            'title': alert.title,
            'message': alert.message,
            'details': alert.details
        }
        
        # Append to daily alert file
        alerts_list = []
        if alert_file.exists():
            try:
                with open(alert_file, 'r') as f:
                    alerts_list = json.load(f)
            except Exception:
                alerts_list = []
        
        alerts_list.append(alert_data)
        
        with open(alert_file, 'w') as f:
            json.dump(alerts_list, f, indent=2, default=str)
    
    def _send_email_alert(self, alert: Alert) -> None:
        """Send alert via email."""
        if not self.email_config['to_emails']:
            self.logger.warning("No email recipients configured for alerts")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] Pipeline Alert: {alert.title}"
            
            body = alert.message
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to SMTP server and send
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            if self.email_config['username']:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent for {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert) -> None:
        """Send alert via webhook."""
        # Placeholder for webhook implementation
        self.logger.info(f"Webhook alert would be sent for {alert.alert_id}")
    
    def _send_dashboard_alert(self, alert: Alert) -> None:
        """Send alert to dashboard."""
        # Placeholder for dashboard alert implementation
        self.logger.info(f"Dashboard alert would be sent for {alert.alert_id}")
    
    def _check_alert_resolution(self, metrics: Dict[str, Any]) -> None:
        """Check if any active alerts can be resolved."""
        for alert_id, alert in list(self.active_alerts.items()):
            if alert.resolved:
                continue
            
            rule = self.alert_rules.get(alert.rule_id)
            if not rule:
                continue
            
            # Check if the condition is no longer true
            try:
                if not rule.condition_function(metrics):
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    
                    # Remove from active alerts
                    del self.active_alerts[alert_id]
                    
                    self.logger.info(f"Alert resolved: {alert.title} (ID: {alert_id})")
                    
                    # Send resolution notification
                    self._send_resolution_notification(alert)
            
            except Exception as e:
                self.logger.error(f"Error checking alert resolution for {alert_id}: {e}")
    
    def _send_resolution_notification(self, alert: Alert) -> None:
        """Send alert resolution notification."""
        resolution_message = (
            f"RESOLVED: {alert.title}\n"
            f"Alert ID: {alert.alert_id}\n"
            f"Triggered: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Resolved: {alert.resolution_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Duration: {(alert.resolution_time - alert.timestamp).total_seconds():.0f} seconds"
        )
        
        self.logger.info(f"ALERT RESOLVED: {resolution_message}")
    
    # Alert condition implementations
    def _check_high_memory_usage(self, metrics: Dict[str, Any]) -> bool:
        """Check for high memory usage."""
        system_metrics = metrics.get('system', {})
        memory_percent = system_metrics.get('memory_percent', 0)
        return memory_percent > 85.0
    
    def _check_critical_memory_usage(self, metrics: Dict[str, Any]) -> bool:
        """Check for critical memory usage."""
        system_metrics = metrics.get('system', {})
        memory_percent = system_metrics.get('memory_percent', 0)
        return memory_percent > 95.0
    
    def _check_high_cpu_usage(self, metrics: Dict[str, Any]) -> bool:
        """Check for high CPU usage."""
        system_metrics = metrics.get('system', {})
        cpu_percent = system_metrics.get('cpu_percent', 0)
        return cpu_percent > 90.0
    
    def _check_low_throughput(self, metrics: Dict[str, Any]) -> bool:
        """Check for low processing throughput."""
        processing_metrics = metrics.get('processing', {})
        throughput = processing_metrics.get('average_throughput', float('inf'))
        return throughput < 1000.0  # Less than 1000 records/sec
    
    def _check_high_error_rate(self, metrics: Dict[str, Any]) -> bool:
        """Check for high error rate."""
        processing_metrics = metrics.get('processing', {})
        success_rate = processing_metrics.get('success_rate', 100.0)
        return success_rate < 90.0  # Less than 90% success rate
    
    def _check_data_quality_degradation(self, metrics: Dict[str, Any]) -> bool:
        """Check for data quality degradation."""
        quality_metrics = metrics.get('data_quality', {})
        error_rate = quality_metrics.get('overall_error_rate', 0.0)
        return error_rate > 10.0  # More than 10% error rate
    
    def _check_pipeline_failure(self, metrics: Dict[str, Any]) -> bool:
        """Check for pipeline failure."""
        health_metrics = metrics.get('health', {})
        overall_status = health_metrics.get('overall_status', 'healthy')
        return overall_status == 'failed'
    
    def _check_low_disk_space(self, metrics: Dict[str, Any]) -> bool:
        """Check for low disk space."""
        system_metrics = metrics.get('system', {})
        disk_usage = system_metrics.get('disk_usage', {})
        
        for path, usage in disk_usage.items():
            if usage.get('percent', 0) > 90.0:  # More than 90% used
                return True
        
        return False
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        if not recent_alerts:
            return {'message': 'No alerts in the specified time period'}
        
        # Count by severity
        by_severity = {}
        for severity in AlertSeverity:
            count = sum(1 for alert in recent_alerts if alert.severity == severity)
            if count > 0:
                by_severity[severity.value] = count
        
        # Count by rule
        by_rule = {}
        for alert in recent_alerts:
            rule_id = alert.rule_id
            if rule_id not in by_rule:
                by_rule[rule_id] = 0
            by_rule[rule_id] += 1
        
        resolved_count = sum(1 for alert in recent_alerts if alert.resolved)
        
        return {
            'time_period_hours': hours,
            'total_alerts': len(recent_alerts),
            'active_alerts': len(self.active_alerts),
            'resolved_alerts': resolved_count,
            'by_severity': by_severity,
            'by_rule': by_rule,
            'resolution_rate': (resolved_count / len(recent_alerts)) * 100,
            'recent_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'rule_id': alert.rule_id,
                    'timestamp': alert.timestamp.isoformat(),
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'resolved': alert.resolved,
                    'acknowledged': alert.acknowledged
                }
                for alert in sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }
    
    def export_alert_report(self, filepath: str, hours: int = 24) -> None:
        """Export alert report to file."""
        summary = self.get_alert_summary(hours)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'alert_summary': summary,
            'alert_rules': {
                rule_id: {
                    'name': rule.name,
                    'description': rule.description,
                    'severity': rule.severity.value,
                    'channels': [ch.value for ch in rule.channels],
                    'enabled': rule.enabled,
                    'trigger_count': rule.trigger_count,
                    'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
                }
                for rule_id, rule in self.alert_rules.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Alert report exported to {filepath}")


# Global alerting system instance
pipeline_alerting_system = PipelineAlertingSystem()


# Convenience functions
def start_alerting() -> None:
    """Start pipeline alerting system."""
    pipeline_alerting_system.start_monitoring()


def stop_alerting() -> None:
    """Stop pipeline alerting system."""
    pipeline_alerting_system.stop_monitoring()


def register_custom_alert_rule(rule_id: str, name: str, description: str,
                             condition_function: Callable[[Dict[str, Any]], bool],
                             severity: AlertSeverity, channels: List[AlertChannel],
                             cooldown_minutes: int = 15) -> None:
    """Register a custom alert rule."""
    pipeline_alerting_system.register_alert_rule(
        rule_id, name, description, condition_function, severity, channels, cooldown_minutes
    )


def get_active_alerts() -> List[Alert]:
    """Get active alerts."""
    return pipeline_alerting_system.get_active_alerts()


def acknowledge_alert(alert_id: str) -> bool:
    """Acknowledge an alert."""
    return pipeline_alerting_system.acknowledge_alert(alert_id)