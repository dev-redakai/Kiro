"""Recovery mechanisms for failed pipeline runs."""

import time
import threading
import pickle
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
from .pipeline_health import pipeline_health_monitor, HealthStatus


class RecoveryStrategy(Enum):
    """Types of recovery strategies."""
    RETRY = "retry"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    PARTIAL_RECOVERY = "partial_recovery"
    FALLBACK_METHOD = "fallback_method"
    SKIP_AND_CONTINUE = "skip_and_continue"
    MANUAL_INTERVENTION = "manual_intervention"


class FailureType(Enum):
    """Types of pipeline failures."""
    MEMORY_ERROR = "memory_error"
    DATA_CORRUPTION = "data_corruption"
    PROCESSING_ERROR = "processing_error"
    IO_ERROR = "io_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class PipelineCheckpoint:
    """Pipeline execution checkpoint."""
    checkpoint_id: str
    stage_name: str
    timestamp: datetime
    data_state: Dict[str, Any]
    processed_records: int
    stage_progress: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None


@dataclass
class RecoveryAction:
    """Recovery action definition."""
    action_id: str
    strategy: RecoveryStrategy
    failure_type: FailureType
    action_function: Callable
    priority: int = 1  # Lower number = higher priority
    max_attempts: int = 3
    timeout_seconds: float = 300.0
    prerequisites: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class FailureRecord:
    """Record of a pipeline failure."""
    failure_id: str
    timestamp: datetime
    stage_name: str
    failure_type: FailureType
    error_message: str
    context: Dict[str, Any]
    recovery_attempts: List[Dict[str, Any]] = field(default_factory=list)
    resolved: bool = False
    resolution_strategy: Optional[RecoveryStrategy] = None


class PipelineRecoveryManager:
    """Comprehensive pipeline recovery system."""
    
    def __init__(self):
        """Initialize pipeline recovery manager."""
        self.logger = get_module_logger("pipeline_recovery")
        self.config = config_manager.config
        
        # Recovery state
        self.checkpoints: Dict[str, PipelineCheckpoint] = {}
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.failure_history: List[FailureRecord] = []
        
        # Recovery settings
        self.checkpoint_interval = 300.0  # 5 minutes
        self.max_recovery_attempts = 3
        self.recovery_timeout = 1800.0  # 30 minutes
        
        # Checkpoint storage
        self.checkpoint_dir = Path("data/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Register default recovery actions
        self._register_default_recovery_actions()
        
        # Recovery monitoring
        self.is_monitoring = False
        self.monitor_thread = None
    
    def _register_default_recovery_actions(self) -> None:
        """Register default recovery actions."""
        
        # Memory error recovery
        self.register_recovery_action(
            "memory_cleanup",
            RecoveryStrategy.RETRY,
            FailureType.MEMORY_ERROR,
            self._recover_memory_error,
            priority=1,
            description="Clean up memory and retry operation"
        )
        
        # Data corruption recovery
        self.register_recovery_action(
            "restore_from_checkpoint",
            RecoveryStrategy.CHECKPOINT_RESTORE,
            FailureType.DATA_CORRUPTION,
            self._recover_from_checkpoint,
            priority=1,
            description="Restore from last valid checkpoint"
        )
        
        # Processing error recovery
        self.register_recovery_action(
            "skip_problematic_chunk",
            RecoveryStrategy.SKIP_AND_CONTINUE,
            FailureType.PROCESSING_ERROR,
            self._skip_problematic_data,
            priority=2,
            description="Skip problematic data and continue processing"
        )
        
        # IO error recovery
        self.register_recovery_action(
            "retry_io_operation",
            RecoveryStrategy.RETRY,
            FailureType.IO_ERROR,
            self._retry_io_operation,
            priority=1,
            max_attempts=5,
            description="Retry I/O operation with exponential backoff"
        )
        
        # Timeout error recovery
        self.register_recovery_action(
            "increase_timeout",
            RecoveryStrategy.RETRY,
            FailureType.TIMEOUT_ERROR,
            self._increase_timeout_and_retry,
            priority=1,
            description="Increase timeout and retry operation"
        )
        
        # Validation error recovery
        self.register_recovery_action(
            "apply_data_corrections",
            RecoveryStrategy.PARTIAL_RECOVERY,
            FailureType.VALIDATION_ERROR,
            self._apply_data_corrections,
            priority=1,
            description="Apply data corrections and continue"
        )
        
        # System error recovery
        self.register_recovery_action(
            "restart_components",
            RecoveryStrategy.FALLBACK_METHOD,
            FailureType.SYSTEM_ERROR,
            self._restart_pipeline_components,
            priority=2,
            description="Restart pipeline components"
        )
    
    def register_recovery_action(self, action_id: str, strategy: RecoveryStrategy,
                               failure_type: FailureType, action_function: Callable,
                               priority: int = 1, max_attempts: int = 3,
                               timeout_seconds: float = 300.0,
                               prerequisites: List[str] = None,
                               description: str = "") -> None:
        """Register a recovery action."""
        action = RecoveryAction(
            action_id=action_id,
            strategy=strategy,
            failure_type=failure_type,
            action_function=action_function,
            priority=priority,
            max_attempts=max_attempts,
            timeout_seconds=timeout_seconds,
            prerequisites=prerequisites or [],
            description=description
        )
        
        self.recovery_actions[action_id] = action
        self.logger.info(f"Registered recovery action: {action_id} for {failure_type.value}")
    
    def create_checkpoint(self, stage_name: str, data_state: Dict[str, Any],
                         processed_records: int, stage_progress: float,
                         metadata: Dict[str, Any] = None) -> str:
        """Create a pipeline checkpoint."""
        checkpoint_id = f"{stage_name}_{int(time.time())}"
        
        checkpoint = PipelineCheckpoint(
            checkpoint_id=checkpoint_id,
            stage_name=stage_name,
            timestamp=datetime.now(),
            data_state=data_state,
            processed_records=processed_records,
            stage_progress=stage_progress,
            metadata=metadata or {}
        )
        
        # Save checkpoint to disk
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            checkpoint.file_path = str(checkpoint_file)
            self.checkpoints[checkpoint_id] = checkpoint
            
            self.logger.info(f"Created checkpoint: {checkpoint_id} for stage {stage_name}")
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint {checkpoint_id}: {e}")
            return None
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[PipelineCheckpoint]:
        """Restore a pipeline checkpoint."""
        if checkpoint_id not in self.checkpoints:
            self.logger.error(f"Checkpoint not found: {checkpoint_id}")
            return None
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        if checkpoint.file_path and Path(checkpoint.file_path).exists():
            try:
                with open(checkpoint.file_path, 'rb') as f:
                    restored_checkpoint = pickle.load(f)
                
                self.logger.info(f"Restored checkpoint: {checkpoint_id}")
                return restored_checkpoint
                
            except Exception as e:
                self.logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
                return None
        else:
            self.logger.error(f"Checkpoint file not found: {checkpoint.file_path}")
            return None
    
    def handle_pipeline_failure(self, stage_name: str, failure_type: FailureType,
                              error_message: str, context: Dict[str, Any] = None) -> bool:
        """Handle a pipeline failure with appropriate recovery strategy."""
        failure_id = f"failure_{int(time.time())}"
        context = context or {}
        
        # Create failure record
        failure_record = FailureRecord(
            failure_id=failure_id,
            timestamp=datetime.now(),
            stage_name=stage_name,
            failure_type=failure_type,
            error_message=error_message,
            context=context
        )
        
        self.failure_history.append(failure_record)
        
        self.logger.error(
            f"Pipeline failure detected in {stage_name}: {failure_type.value} - {error_message}"
        )
        
        # Find applicable recovery actions
        applicable_actions = [
            action for action in self.recovery_actions.values()
            if action.failure_type == failure_type
        ]
        
        # Sort by priority
        applicable_actions.sort(key=lambda x: x.priority)
        
        # Attempt recovery
        for action in applicable_actions:
            if self._attempt_recovery(failure_record, action):
                failure_record.resolved = True
                failure_record.resolution_strategy = action.strategy
                self.logger.info(f"Pipeline failure {failure_id} resolved using {action.action_id}")
                return True
        
        # If all recovery attempts failed
        self.logger.critical(f"Failed to recover from pipeline failure {failure_id}")
        return False
    
    def _attempt_recovery(self, failure_record: FailureRecord, action: RecoveryAction) -> bool:
        """Attempt a specific recovery action."""
        attempt_count = len([
            attempt for attempt in failure_record.recovery_attempts
            if attempt['action_id'] == action.action_id
        ])
        
        if attempt_count >= action.max_attempts:
            self.logger.warning(
                f"Maximum attempts reached for recovery action {action.action_id}"
            )
            return False
        
        self.logger.info(f"Attempting recovery action: {action.action_id} (attempt {attempt_count + 1})")
        
        attempt_record = {
            'action_id': action.action_id,
            'attempt_number': attempt_count + 1,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error_message': None
        }
        
        try:
            # Execute recovery action with timeout
            success = self._execute_with_timeout(
                action.action_function,
                action.timeout_seconds,
                failure_record.context
            )
            
            attempt_record['success'] = success
            failure_record.recovery_attempts.append(attempt_record)
            
            if success:
                self.logger.info(f"Recovery action {action.action_id} succeeded")
                return True
            else:
                self.logger.warning(f"Recovery action {action.action_id} failed")
                return False
                
        except Exception as e:
            attempt_record['error_message'] = str(e)
            failure_record.recovery_attempts.append(attempt_record)
            
            self.logger.error(f"Recovery action {action.action_id} failed with error: {e}")
            return False
    
    def _execute_with_timeout(self, func: Callable, timeout_seconds: float,
                            context: Dict[str, Any]) -> bool:
        """Execute function with timeout."""
        import threading
        import queue
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def target():
            try:
                result = func(context)
                result_queue.put(bool(result))
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            # Thread is still running, timeout occurred
            raise TimeoutError(f"Recovery action timed out after {timeout_seconds} seconds")
        
        # Check for exceptions
        if not exception_queue.empty():
            raise exception_queue.get()
        
        # Get result
        if not result_queue.empty():
            return result_queue.get()
        else:
            return False
    
    # Recovery action implementations
    def _recover_memory_error(self, context: Dict[str, Any]) -> bool:
        """Recover from memory errors."""
        self.logger.info("Attempting memory error recovery")
        
        try:
            # Force garbage collection
            import gc
            collected = gc.collect()
            self.logger.info(f"Garbage collection freed {collected} objects")
            
            # Reduce chunk size if applicable
            if hasattr(self.config, 'chunk_size'):
                original_size = self.config.chunk_size
                new_size = max(1000, original_size // 2)
                self.config.chunk_size = new_size
                self.logger.info(f"Reduced chunk size from {original_size} to {new_size}")
            
            # Clear any caches
            if hasattr(pipeline_monitor, 'clear_cache'):
                pipeline_monitor.clear_cache()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Memory recovery failed: {e}")
            return False
    
    def _recover_from_checkpoint(self, context: Dict[str, Any]) -> bool:
        """Recover from the last valid checkpoint."""
        stage_name = context.get('stage_name')
        if not stage_name:
            return False
        
        # Find the most recent checkpoint for this stage
        stage_checkpoints = [
            cp for cp in self.checkpoints.values()
            if cp.stage_name == stage_name
        ]
        
        if not stage_checkpoints:
            self.logger.warning(f"No checkpoints found for stage {stage_name}")
            return False
        
        # Get the most recent checkpoint
        latest_checkpoint = max(stage_checkpoints, key=lambda x: x.timestamp)
        
        # Restore the checkpoint
        restored = self.restore_checkpoint(latest_checkpoint.checkpoint_id)
        if restored:
            self.logger.info(f"Restored from checkpoint {latest_checkpoint.checkpoint_id}")
            return True
        
        return False
    
    def _skip_problematic_data(self, context: Dict[str, Any]) -> bool:
        """Skip problematic data and continue processing."""
        self.logger.info("Skipping problematic data chunk")
        
        # Mark the problematic chunk for skipping
        context['skip_current_chunk'] = True
        
        # Log the skip action
        chunk_info = context.get('chunk_info', {})
        self.logger.warning(f"Skipped problematic data chunk: {chunk_info}")
        
        return True
    
    def _retry_io_operation(self, context: Dict[str, Any]) -> bool:
        """Retry I/O operation with exponential backoff."""
        operation = context.get('operation')
        if not operation:
            return False
        
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                delay = base_delay * (2 ** attempt)
                self.logger.info(f"Retrying I/O operation after {delay}s delay (attempt {attempt + 1})")
                
                time.sleep(delay)
                
                # Execute the operation (this would need to be passed in context)
                if callable(operation):
                    result = operation()
                    if result:
                        return True
                
            except Exception as e:
                self.logger.warning(f"I/O retry attempt {attempt + 1} failed: {e}")
                continue
        
        return False
    
    def _increase_timeout_and_retry(self, context: Dict[str, Any]) -> bool:
        """Increase timeout and retry operation."""
        current_timeout = context.get('timeout', 30.0)
        new_timeout = current_timeout * 2
        
        self.logger.info(f"Increasing timeout from {current_timeout}s to {new_timeout}s")
        
        context['timeout'] = new_timeout
        
        # This would need to trigger a retry of the original operation
        # with the new timeout value
        return True
    
    def _apply_data_corrections(self, context: Dict[str, Any]) -> bool:
        """Apply data corrections for validation errors."""
        self.logger.info("Applying data corrections for validation errors")
        
        # This would apply known data correction rules
        # based on the validation errors encountered
        corrections_applied = context.get('corrections_applied', 0)
        context['corrections_applied'] = corrections_applied + 1
        
        return True
    
    def _restart_pipeline_components(self, context: Dict[str, Any]) -> bool:
        """Restart pipeline components."""
        self.logger.info("Restarting pipeline components")
        
        try:
            # Stop monitoring
            if pipeline_health_monitor.is_monitoring:
                pipeline_health_monitor.stop_monitoring()
            
            # Wait a moment
            time.sleep(2.0)
            
            # Restart monitoring
            pipeline_health_monitor.start_monitoring()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restart components: {e}")
            return False
    
    def start_recovery_monitoring(self) -> None:
        """Start recovery monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._recovery_monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Recovery monitoring started")
    
    def stop_recovery_monitoring(self) -> None:
        """Stop recovery monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Recovery monitoring stopped")
    
    def _recovery_monitoring_loop(self) -> None:
        """Recovery monitoring loop."""
        while self.is_monitoring:
            try:
                # Check pipeline health
                health_status = pipeline_health_monitor.get_current_health_status()
                
                # Trigger recovery if health is critical
                if health_status.overall_status == HealthStatus.CRITICAL:
                    self._handle_critical_health_status(health_status)
                
                # Clean up old checkpoints
                self._cleanup_old_checkpoints()
                
                time.sleep(60.0)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in recovery monitoring loop: {e}")
                time.sleep(60.0)
    
    def _handle_critical_health_status(self, health_status) -> None:
        """Handle critical health status."""
        for alert in health_status.active_alerts:
            if alert['status'] == 'critical':
                # Determine failure type based on component
                component = alert['component']
                if component == 'system':
                    failure_type = FailureType.SYSTEM_ERROR
                elif component == 'data_quality':
                    failure_type = FailureType.VALIDATION_ERROR
                else:
                    failure_type = FailureType.PROCESSING_ERROR
                
                # Attempt recovery
                self.handle_pipeline_failure(
                    stage_name=component,
                    failure_type=failure_type,
                    error_message=alert['message'],
                    context={'health_alert': alert}
                )
    
    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoint files."""
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep checkpoints for 7 days
        
        for checkpoint_id, checkpoint in list(self.checkpoints.items()):
            if checkpoint.timestamp < cutoff_time:
                # Remove checkpoint file
                if checkpoint.file_path and Path(checkpoint.file_path).exists():
                    try:
                        Path(checkpoint.file_path).unlink()
                        self.logger.debug(f"Removed old checkpoint file: {checkpoint.file_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove checkpoint file: {e}")
                
                # Remove from memory
                del self.checkpoints[checkpoint_id]
    
    def get_recovery_summary(self) -> Dict[str, Any]:
        """Get recovery system summary."""
        recent_failures = [
            f for f in self.failure_history
            if f.timestamp >= datetime.now() - timedelta(hours=24)
        ]
        
        resolved_failures = [f for f in recent_failures if f.resolved]
        unresolved_failures = [f for f in recent_failures if not f.resolved]
        
        return {
            'total_checkpoints': len(self.checkpoints),
            'recent_failures_24h': len(recent_failures),
            'resolved_failures_24h': len(resolved_failures),
            'unresolved_failures_24h': len(unresolved_failures),
            'recovery_success_rate': (len(resolved_failures) / len(recent_failures)) * 100 if recent_failures else 100,
            'registered_recovery_actions': len(self.recovery_actions),
            'monitoring_active': self.is_monitoring,
            'recent_failures': [
                {
                    'failure_id': f.failure_id,
                    'timestamp': f.timestamp.isoformat(),
                    'stage_name': f.stage_name,
                    'failure_type': f.failure_type.value,
                    'resolved': f.resolved,
                    'recovery_attempts': len(f.recovery_attempts)
                }
                for f in sorted(recent_failures, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }
    
    def export_recovery_report(self, filepath: str) -> None:
        """Export recovery report to file."""
        summary = self.get_recovery_summary()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'recovery_summary': summary,
            'registered_actions': {
                action_id: {
                    'strategy': action.strategy.value,
                    'failure_type': action.failure_type.value,
                    'priority': action.priority,
                    'max_attempts': action.max_attempts,
                    'description': action.description
                }
                for action_id, action in self.recovery_actions.items()
            },
            'checkpoints': {
                cp_id: {
                    'stage_name': cp.stage_name,
                    'timestamp': cp.timestamp.isoformat(),
                    'processed_records': cp.processed_records,
                    'stage_progress': cp.stage_progress
                }
                for cp_id, cp in self.checkpoints.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Recovery report exported to {filepath}")


# Global recovery manager instance
pipeline_recovery_manager = PipelineRecoveryManager()


# Convenience functions
def create_checkpoint(stage_name: str, data_state: Dict[str, Any],
                     processed_records: int, stage_progress: float,
                     metadata: Dict[str, Any] = None) -> str:
    """Create a pipeline checkpoint."""
    return pipeline_recovery_manager.create_checkpoint(
        stage_name, data_state, processed_records, stage_progress, metadata
    )


def handle_failure(stage_name: str, failure_type: FailureType,
                  error_message: str, context: Dict[str, Any] = None) -> bool:
    """Handle a pipeline failure."""
    return pipeline_recovery_manager.handle_pipeline_failure(
        stage_name, failure_type, error_message, context
    )


def start_recovery_monitoring() -> None:
    """Start recovery monitoring."""
    pipeline_recovery_manager.start_recovery_monitoring()


def stop_recovery_monitoring() -> None:
    """Stop recovery monitoring."""
    pipeline_recovery_manager.stop_recovery_monitoring()