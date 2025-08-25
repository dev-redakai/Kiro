"""CLI commands for monitoring and health checks."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .monitoring_dashboard import get_dashboard_data, get_health_status, export_monitoring_report
from .pipeline_health import pipeline_health_monitor, start_health_monitoring, stop_health_monitoring
from .pipeline_alerting import pipeline_alerting_system, start_alerting, stop_alerting
from .pipeline_recovery import pipeline_recovery_manager, start_recovery_monitoring, stop_recovery_monitoring
from .monitoring import pipeline_monitor, start_monitoring, stop_monitoring
from .logger import get_module_logger


def create_monitoring_cli() -> argparse.ArgumentParser:
    """Create monitoring CLI parser."""
    parser = argparse.ArgumentParser(
        description="Pipeline Monitoring and Health Check CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.utils.monitoring_cli status
  python -m src.utils.monitoring_cli health
  python -m src.utils.monitoring_cli start-monitoring
  python -m src.utils.monitoring_cli export-report --output monitoring_report.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show pipeline status')
    status_parser.add_argument('--format', choices=['json', 'table'], default='table',
                              help='Output format')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Show health status')
    health_parser.add_argument('--format', choices=['json', 'table'], default='table',
                              help='Output format')
    
    # Start monitoring command
    start_parser = subparsers.add_parser('start-monitoring', help='Start all monitoring systems')
    start_parser.add_argument('--components', nargs='+', 
                             choices=['health', 'alerting', 'recovery', 'monitoring'],
                             default=['health', 'alerting', 'recovery', 'monitoring'],
                             help='Components to start')
    
    # Stop monitoring command
    stop_parser = subparsers.add_parser('stop-monitoring', help='Stop all monitoring systems')
    stop_parser.add_argument('--components', nargs='+',
                            choices=['health', 'alerting', 'recovery', 'monitoring'],
                            default=['health', 'alerting', 'recovery', 'monitoring'],
                            help='Components to stop')
    
    # Export report command
    export_parser = subparsers.add_parser('export-report', help='Export monitoring report')
    export_parser.add_argument('--output', '-o', required=True,
                              help='Output file path')
    export_parser.add_argument('--format', choices=['json'], default='json',
                              help='Export format')
    
    # Dashboard data command
    dashboard_parser = subparsers.add_parser('dashboard', help='Get dashboard data')
    dashboard_parser.add_argument('--refresh', action='store_true',
                                 help='Force refresh of cached data')
    
    # Alerts command
    alerts_parser = subparsers.add_parser('alerts', help='Show active alerts')
    alerts_parser.add_argument('--format', choices=['json', 'table'], default='table',
                              help='Output format')
    
    return parser


def format_status_table(data: dict) -> str:
    """Format status data as table."""
    lines = []
    lines.append("Pipeline Status")
    lines.append("=" * 50)
    
    system_status = data.get('system_status', {})
    lines.append(f"Overall Status: {system_status.get('overall_status', 'unknown').upper()}")
    lines.append(f"Healthy Components: {system_status.get('healthy_components', 0)}")
    lines.append(f"Warning Components: {system_status.get('warning_components', 0)}")
    lines.append(f"Critical Components: {system_status.get('critical_components', 0)}")
    lines.append(f"Last Updated: {system_status.get('last_updated', 'unknown')}")
    
    lines.append("\nPerformance Metrics")
    lines.append("-" * 30)
    perf_metrics = data.get('performance_metrics', {})
    lines.append(f"Total Operations: {perf_metrics.get('total_operations', 0)}")
    lines.append(f"Success Rate: {perf_metrics.get('success_rate', 0):.1f}%")
    lines.append(f"Average Throughput: {perf_metrics.get('average_throughput', 0):.2f} records/sec")
    lines.append(f"Total Records Processed: {perf_metrics.get('total_records_processed', 0):,}")
    
    lines.append("\nResource Usage")
    lines.append("-" * 30)
    resource_usage = data.get('resource_usage', {})
    memory = resource_usage.get('memory', {})
    cpu = resource_usage.get('cpu', {})
    lines.append(f"Memory Usage: {memory.get('percent', 0):.1f}% ({memory.get('used_gb', 0):.2f}GB / {memory.get('total_gb', 0):.2f}GB)")
    lines.append(f"CPU Usage: {cpu.get('percent', 0):.1f}%")
    
    return "\n".join(lines)


def format_health_table(data: dict) -> str:
    """Format health data as table."""
    lines = []
    lines.append("Pipeline Health Status")
    lines.append("=" * 50)
    
    pipeline_health = data.get('pipeline_health', {})
    lines.append(f"Current Status: {pipeline_health.get('current_status', 'unknown').upper()}")
    lines.append(f"Active Alerts: {pipeline_health.get('active_alerts_count', 0)}")
    lines.append(f"Last Check: {pipeline_health.get('last_check', 'unknown')}")
    
    lines.append("\nComponent Status")
    lines.append("-" * 30)
    component_statuses = pipeline_health.get('component_statuses', {})
    for component, status_info in component_statuses.items():
        status = status_info.get('status', 'unknown')
        lines.append(f"{component.replace('_', ' ').title()}: {status.upper()}")
    
    return "\n".join(lines)


def format_alerts_table(data: dict) -> str:
    """Format alerts data as table."""
    lines = []
    lines.append("Active Alerts")
    lines.append("=" * 50)
    
    active_alerts = data.get('active_alerts', {})
    total_alerts = active_alerts.get('total_active_alerts', 0)
    lines.append(f"Total Active Alerts: {total_alerts}")
    
    if total_alerts > 0:
        lines.append("\nRecent Alerts")
        lines.append("-" * 30)
        recent_alerts = active_alerts.get('recent_alerts', [])
        for alert in recent_alerts[:10]:
            timestamp = alert.get('timestamp', '')[:19]  # Remove microseconds
            lines.append(f"[{alert.get('severity', '').upper()}] {alert.get('title', '')} ({timestamp})")
    else:
        lines.append("\nNo active alerts")
    
    return "\n".join(lines)


def cmd_status(args):
    """Handle status command."""
    logger = get_module_logger("monitoring_cli")
    
    try:
        data = get_dashboard_data()
        
        if args.format == 'json':
            print(json.dumps(data, indent=2, default=str))
        else:
            print(format_status_table(data))
            
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def cmd_health(args):
    """Handle health command."""
    logger = get_module_logger("monitoring_cli")
    
    try:
        if args.format == 'json':
            health_data = get_health_status()
            print(json.dumps(health_data, indent=2, default=str))
        else:
            dashboard_data = get_dashboard_data()
            print(format_health_table(dashboard_data))
            
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def cmd_start_monitoring(args):
    """Handle start monitoring command."""
    logger = get_module_logger("monitoring_cli")
    
    try:
        components_started = []
        
        if 'monitoring' in args.components:
            start_monitoring()
            components_started.append('monitoring')
            
        if 'health' in args.components:
            start_health_monitoring()
            components_started.append('health')
            
        if 'alerting' in args.components:
            start_alerting()
            components_started.append('alerting')
            
        if 'recovery' in args.components:
            start_recovery_monitoring()
            components_started.append('recovery')
        
        print(f"Started monitoring components: {', '.join(components_started)}")
        logger.info(f"Started monitoring components via CLI: {components_started}")
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def cmd_stop_monitoring(args):
    """Handle stop monitoring command."""
    logger = get_module_logger("monitoring_cli")
    
    try:
        components_stopped = []
        
        if 'monitoring' in args.components:
            stop_monitoring()
            components_stopped.append('monitoring')
            
        if 'health' in args.components:
            stop_health_monitoring()
            components_stopped.append('health')
            
        if 'alerting' in args.components:
            stop_alerting()
            components_stopped.append('alerting')
            
        if 'recovery' in args.components:
            stop_recovery_monitoring()
            components_stopped.append('recovery')
        
        print(f"Stopped monitoring components: {', '.join(components_stopped)}")
        logger.info(f"Stopped monitoring components via CLI: {components_stopped}")
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def cmd_export_report(args):
    """Handle export report command."""
    logger = get_module_logger("monitoring_cli")
    
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_monitoring_report(str(output_path))
        print(f"Monitoring report exported to: {output_path}")
        logger.info(f"Exported monitoring report to {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting report: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def cmd_dashboard(args):
    """Handle dashboard command."""
    logger = get_module_logger("monitoring_cli")
    
    try:
        data = get_dashboard_data(force_refresh=args.refresh)
        print(json.dumps(data, indent=2, default=str))
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def cmd_alerts(args):
    """Handle alerts command."""
    logger = get_module_logger("monitoring_cli")
    
    try:
        data = get_dashboard_data()
        
        if args.format == 'json':
            alerts_data = data.get('active_alerts', {})
            print(json.dumps(alerts_data, indent=2, default=str))
        else:
            print(format_alerts_table(data))
            
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = create_monitoring_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Command dispatch
    commands = {
        'status': cmd_status,
        'health': cmd_health,
        'start-monitoring': cmd_start_monitoring,
        'stop-monitoring': cmd_stop_monitoring,
        'export-report': cmd_export_report,
        'dashboard': cmd_dashboard,
        'alerts': cmd_alerts
    }
    
    command_func = commands.get(args.command)
    if command_func:
        return command_func(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())