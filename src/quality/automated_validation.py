"""Automated data quality validation system."""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import json
import psutil

from ..utils.logger import get_module_logger
from ..utils.config import config_manager
from ..utils.monitoring import pipeline_monitor, log_data_quality_metric
from ..utils.error_handler import error_handler, DataQualityError, ErrorSeverity, ErrorCategory
import json


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationRule(Enum):
    """Types of validation rules."""
    NOT_NULL = "not_null"
    UNIQUE = "unique"
    RANGE = "range"
    FORMAT = "format"
    ENUM = "enum"
    CUSTOM = "custom"
    STATISTICAL = "statistical"
    BUSINESS_LOGIC = "business_logic"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_name: str
    field_name: str
    rule_type: ValidationRule
    severity: ValidationSeverity
    passed: bool
    total_records: int
    valid_records: int
    invalid_records: int
    error_rate: float
    details: Dict[str, Any] = field(default_factory=dict)
    sample_invalid_values: List[Any] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationExecutionContext:
    """Context information for validation rule execution."""
    rule_name: str
    field_name: str
    data_type: str
    sample_values: List[Any]
    error_category: ErrorCategory
    recovery_attempted: bool = False
    execution_time: float = 0.0
    fallback_applied: bool = False
    problematic_values: List[Any] = field(default_factory=list)
    validation_parameters: Dict[str, Any] = field(default_factory=dict)
    error_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationErrorReport:
    """Detailed error report for validation failures."""
    rule_name: str
    field_name: str
    error_type: str
    error_message: str
    problematic_values: List[Any]
    sample_size: int
    total_errors: int
    error_percentage: float
    suggested_actions: List[str]
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationMetrics:
    """Performance and quality metrics for validation execution."""
    total_rules_executed: int
    successful_rules: int
    failed_rules: int
    skipped_rules: int
    total_execution_time: float
    average_rule_time: float
    total_records_validated: int
    total_valid_records: int
    total_invalid_records: int
    overall_data_quality_score: float
    rule_performance: Dict[str, float]
    error_distribution: Dict[str, int]
    memory_usage_mb: float


@dataclass
class DataQualityRule:
    """Data quality validation rule definition."""
    name: str
    field_name: str
    rule_type: ValidationRule
    severity: ValidationSeverity
    validation_function: Callable[[pd.Series], Tuple[pd.Series, Dict[str, Any]]]
    enabled: bool = True
    threshold: Optional[float] = None  # Error rate threshold
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


class AutomatedDataValidator:
    """Automated data quality validation system."""
    
    def __init__(self):
        """Initialize automated data validator."""
        self.logger = get_module_logger("automated_validator")
        self.config = config_manager.config
        
        # Validation rules registry
        self.validation_rules: Dict[str, DataQualityRule] = {}
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
        
        # Enhanced error reporting
        self.error_reports: List[ValidationErrorReport] = []
        self.validation_metrics: Optional[ValidationMetrics] = None
        
        # Quality thresholds
        self.default_error_thresholds = {
            ValidationSeverity.INFO: 1.0,      # 1%
            ValidationSeverity.WARNING: 5.0,   # 5%
            ValidationSeverity.ERROR: 10.0,    # 10%
            ValidationSeverity.CRITICAL: 20.0  # 20%
        }
        
        # Register default validation rules
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """Register default validation rules for e-commerce data."""
        
        # Order ID validation
        self.register_validation_rule(
            "order_id_not_null",
            "order_id",
            ValidationRule.NOT_NULL,
            ValidationSeverity.CRITICAL,
            self._validate_not_null,
            description="Order ID must not be null"
        )
        
        self.register_validation_rule(
            "order_id_unique",
            "order_id",
            ValidationRule.UNIQUE,
            ValidationSeverity.ERROR,
            self._validate_unique,
            description="Order ID should be unique"
        )
        
        # Product name validation
        self.register_validation_rule(
            "product_name_not_null",
            "product_name",
            ValidationRule.NOT_NULL,
            ValidationSeverity.ERROR,
            self._validate_not_null,
            description="Product name must not be null"
        )
        
        self.register_validation_rule(
            "product_name_length",
            "product_name",
            ValidationRule.RANGE,
            ValidationSeverity.WARNING,
            self._validate_string_length,
            parameters={'min_length': 1, 'max_length': 200},
            description="Product name length should be between 1 and 200 characters"
        )
        
        # Category validation
        self.register_validation_rule(
            "category_enum",
            "category",
            ValidationRule.ENUM,
            ValidationSeverity.WARNING,
            self._validate_enum,
            parameters={'valid_values': ['Electronics', 'Fashion', 'Home & Garden', 'Unknown']},
            description="Category should be one of the valid values"
        )
        
        # Quantity validation
        self.register_validation_rule(
            "quantity_positive",
            "quantity",
            ValidationRule.RANGE,
            ValidationSeverity.ERROR,
            self._validate_numeric_range,
            parameters={'min_value': 1, 'max_value': 10000},
            description="Quantity should be positive and reasonable"
        )
        
        self.register_validation_rule(
            "quantity_statistical",
            "quantity",
            ValidationRule.STATISTICAL,
            ValidationSeverity.WARNING,
            self._validate_statistical_outliers,
            parameters={'z_score_threshold': 3.0},
            description="Quantity should not be a statistical outlier"
        )
        
        # Unit price validation
        self.register_validation_rule(
            "unit_price_positive",
            "unit_price",
            ValidationRule.RANGE,
            ValidationSeverity.ERROR,
            self._validate_numeric_range,
            parameters={'min_value': 0.01, 'max_value': 100000.0},
            description="Unit price should be positive and reasonable"
        )
        
        # Discount validation
        self.register_validation_rule(
            "discount_range",
            "discount_percent",
            ValidationRule.RANGE,
            ValidationSeverity.ERROR,
            self._validate_numeric_range,
            parameters={'min_value': 0.0, 'max_value': 1.0},
            description="Discount percent should be between 0.0 and 1.0"
        )
        
        # Region validation
        self.register_validation_rule(
            "region_enum",
            "region",
            ValidationRule.ENUM,
            ValidationSeverity.WARNING,
            self._validate_enum,
            parameters={'valid_values': ['North', 'South', 'East', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest', 'Unknown']},
            description="Region should be one of the valid values"
        )
        
        # Email validation
        self.register_validation_rule(
            "email_format",
            "customer_email",
            ValidationRule.FORMAT,
            ValidationSeverity.WARNING,
            self._validate_email_format,
            description="Email should have valid format when present"
        )
        
        # Date validation
        self.register_validation_rule(
            "sale_date_range",
            "sale_date",
            ValidationRule.RANGE,
            ValidationSeverity.ERROR,
            self._validate_date_range,
            parameters={'min_date': '2020-01-01', 'max_date': '2030-12-31'},
            description="Sale date should be within reasonable range"
        )
        
        # Business logic validation
        self.register_validation_rule(
            "revenue_calculation",
            "revenue",
            ValidationRule.BUSINESS_LOGIC,
            ValidationSeverity.ERROR,
            self._validate_revenue_calculation,
            description="Revenue should equal quantity * unit_price * (1 - discount_percent)"
        )
    
    def register_validation_rule(self, name: str, field_name: str, rule_type: ValidationRule,
                               severity: ValidationSeverity, validation_function: Callable,
                               threshold: Optional[float] = None,
                               parameters: Dict[str, Any] = None,
                               description: str = "") -> None:
        """Register a new validation rule."""
        rule = DataQualityRule(
            name=name,
            field_name=field_name,
            rule_type=rule_type,
            severity=severity,
            validation_function=validation_function,
            threshold=threshold or self.default_error_thresholds[severity],
            parameters=parameters or {},
            description=description
        )
        
        self.validation_rules[name] = rule
        self.logger.info(f"Registered validation rule: {name} for field {field_name}")
    
    def validate_dataframe(self, df: pd.DataFrame, 
                         rules_to_run: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate a DataFrame against all or specified rules with enhanced error resilience."""
        if df.empty:
            self.logger.warning("Cannot validate empty DataFrame")
            return {'status': 'skipped', 'reason': 'empty_dataframe'}
        
        self.logger.info(f"Starting automated validation of {len(df)} records")
        start_time = time.time()
        
        # Determine which rules to run
        if rules_to_run is None:
            rules_to_run = list(self.validation_rules.keys())
        
        # Enhanced tracking for resilient validation
        validation_results = []
        rule_execution_stats = {
            'total_rules_attempted': 0,
            'rules_executed_successfully': 0,
            'rules_failed_execution': 0,
            'rules_skipped': 0,
            'execution_errors': []
        }
        
        overall_status = "passed"
        critical_failures = 0
        error_failures = 0
        warning_failures = 0
        
        # Run each validation rule with individual error isolation
        for rule_name in rules_to_run:
            rule_execution_stats['total_rules_attempted'] += 1
            
            # Rule existence check with isolation
            if rule_name not in self.validation_rules:
                self.logger.warning(f"Validation rule not found: {rule_name}")
                rule_execution_stats['rules_skipped'] += 1
                rule_execution_stats['execution_errors'].append({
                    'rule_name': rule_name,
                    'error_type': 'rule_not_found',
                    'error_message': f"Rule '{rule_name}' not registered",
                    'severity': 'warning'
                })
                continue
            
            rule = self.validation_rules[rule_name]
            
            # Rule enabled check with isolation
            if not rule.enabled:
                self.logger.debug(f"Validation rule {rule_name} is disabled, skipping")
                rule_execution_stats['rules_skipped'] += 1
                continue
            
            # Field existence check with isolation
            if rule.field_name not in df.columns:
                self.logger.warning(f"Field {rule.field_name} not found in DataFrame for rule {rule_name}")
                rule_execution_stats['rules_skipped'] += 1
                rule_execution_stats['execution_errors'].append({
                    'rule_name': rule_name,
                    'field_name': rule.field_name,
                    'error_type': 'field_not_found',
                    'error_message': f"Field '{rule.field_name}' not found in DataFrame",
                    'severity': 'warning'
                })
                continue
            
            # Execute validation rule with comprehensive error isolation
            try:
                self.logger.debug(f"Executing validation rule: {rule_name}")
                # Pass execution stats for error tracking
                self._current_execution_stats = rule_execution_stats
                result = self._run_validation_rule_with_isolation(df, rule)
                
                if result is not None:
                    validation_results.append(result)
                    rule_execution_stats['rules_executed_successfully'] += 1
                    
                    # Update overall status based on validation result
                    if not result.passed:
                        if result.severity == ValidationSeverity.CRITICAL:
                            critical_failures += 1
                            overall_status = "critical"
                        elif result.severity == ValidationSeverity.ERROR:
                            error_failures += 1
                            if overall_status not in ["critical"]:
                                overall_status = "error"
                        elif result.severity == ValidationSeverity.WARNING:
                            warning_failures += 1
                            if overall_status not in ["critical", "error"]:
                                overall_status = "warning"
                    
                    # Log validation result
                    self.logger.info(
                        f"Validation {rule_name}: {'PASSED' if result.passed else 'FAILED'} "
                        f"({result.error_rate:.2f}% error rate)"
                    )
                    
                    # Log to monitoring system with error handling
                    try:
                        log_data_quality_metric(
                            rule_name,
                            result.total_records,
                            result.valid_records,
                            result.invalid_records,
                            {'rule_type': result.rule_type.value, 'severity': result.severity.value}
                        )
                    except Exception as monitoring_error:
                        self.logger.warning(f"Failed to log monitoring metric for {rule_name}: {monitoring_error}")
                else:
                    rule_execution_stats['rules_failed_execution'] += 1
                    
            except Exception as e:
                # Comprehensive error handling for rule execution failures
                rule_execution_stats['rules_failed_execution'] += 1
                error_context = {
                    'rule_name': rule_name,
                    'field_name': rule.field_name,
                    'rule_type': rule.rule_type.value,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
                
                rule_execution_stats['execution_errors'].append({
                    'rule_name': rule_name,
                    'field_name': rule.field_name,
                    'error_type': 'execution_failure',
                    'error_message': str(e),
                    'severity': 'error',
                    'exception_type': type(e).__name__
                })
                
                self.logger.error(f"Critical error running validation rule {rule_name}: {e}")
                self.logger.debug(f"Error context: {error_context}")
                
                # Handle error through error handler with proper categorization
                try:
                    error_handler.handle_error(
                        DataQualityError(f"Validation rule execution failed: {rule_name}", rule.field_name),
                        ErrorCategory.VALIDATION,
                        ErrorSeverity.MEDIUM,
                        context=error_context
                    )
                except Exception as handler_error:
                    # Even error handler failures shouldn't crash the validation
                    self.logger.error(f"Error handler failed for rule {rule_name}: {handler_error}")
                    rule_execution_stats['execution_errors'].append({
                        'rule_name': rule_name,
                        'error_type': 'error_handler_failure',
                        'error_message': f"Error handler failed: {handler_error}",
                        'severity': 'critical'
                    })
                
                # Continue with next rule - don't let single rule failure stop entire validation
                continue
        
        # Store validation results with error handling
        try:
            self.validation_history.extend(validation_results)
            
            # Keep only last 1000 validation results
            if len(self.validation_history) > 1000:
                self.validation_history = self.validation_history[-1000:]
        except Exception as history_error:
            self.logger.warning(f"Failed to store validation history: {history_error}")
        
        duration = time.time() - start_time
        
        # Generate comprehensive validation summary with success/failure metrics
        summary = self._generate_comprehensive_validation_summary(
            validation_results, rule_execution_stats, overall_status,
            critical_failures, error_failures, warning_failures,
            len(df), duration
        )
        
        # Enhanced logging with comprehensive metrics
        self._log_validation_metrics(validation_results, rule_execution_stats, duration)
        
        # Log problematic values analysis
        self._log_problematic_values_analysis(validation_results)
        
        # Log comprehensive completion summary
        success_rate = (rule_execution_stats['rules_executed_successfully'] / 
                       max(1, rule_execution_stats['total_rules_attempted'])) * 100
        
        self.logger.info(
            f"VALIDATION SESSION COMPLETED in {duration:.2f}s: {overall_status.upper()}\n"
            f"  Status Breakdown: {critical_failures} critical, {error_failures} errors, {warning_failures} warnings\n"
            f"  Rule Execution: {rule_execution_stats['rules_executed_successfully']}/{rule_execution_stats['total_rules_attempted']} "
            f"({success_rate:.1f}% success rate)\n"
            f"  Data Quality: {len(df)} records processed"
        )
        
        # Log execution errors summary if any occurred
        if rule_execution_stats['execution_errors']:
            error_summary = {}
            for error in rule_execution_stats['execution_errors']:
                error_type = error['error_type']
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
            
            self.logger.warning(f"Validation execution errors summary: {error_summary}")
            
            # Log actionable recommendations for common error patterns
            if 'validation_function_failure' in error_summary:
                self.logger.error("CRITICAL: Validation function failures detected - check rule implementations")
            
            if 'field_not_found' in error_summary:
                self.logger.warning("WARNING: Missing fields detected - verify data schema consistency")
            
            if 'error_handler_failure' in error_summary:
                self.logger.error("CRITICAL: Error handler failures detected - check error handling configuration")
        
        return summary
    
    def _log_validation_error_with_context(self, rule: DataQualityRule, error: Exception, 
                                         context: ValidationExecutionContext) -> None:
        """Log validation error with detailed context and actionable information."""
        # Create structured error message
        error_details = {
            'rule_name': rule.name,
            'field_name': rule.field_name,
            'rule_type': rule.rule_type.value,
            'severity': rule.severity.value,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'data_type_detected': context.data_type,
            'sample_values': context.sample_values[:5],  # First 5 sample values
            'problematic_values': context.problematic_values[:10],  # First 10 problematic values
            'validation_parameters': context.validation_parameters,
            'execution_time_ms': context.execution_time * 1000,
            'recovery_attempted': context.recovery_attempted,
            'fallback_applied': context.fallback_applied,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate actionable suggestions
        suggestions = self._generate_actionable_suggestions(rule, error, context)
        error_details['suggested_actions'] = suggestions
        
        # Log structured error with context
        self.logger.error(
            f"VALIDATION ERROR - {rule.name}: {str(error)}\n"
            f"  Field: {rule.field_name} (type: {context.data_type})\n"
            f"  Rule Type: {rule.rule_type.value}\n"
            f"  Severity: {rule.severity.value}\n"
            f"  Execution Time: {context.execution_time*1000:.2f}ms\n"
            f"  Sample Values: {context.sample_values[:3]}\n"
            f"  Problematic Values: {context.problematic_values[:5]}\n"
            f"  Suggested Actions: {', '.join(suggestions[:3])}"
        )
        
        # Log detailed context as JSON for debugging
        self.logger.debug(f"Validation Error Context: {json.dumps(error_details, indent=2, default=str)}")
        
        # Create error report
        error_report = ValidationErrorReport(
            rule_name=rule.name,
            field_name=rule.field_name,
            error_type=type(error).__name__,
            error_message=str(error),
            problematic_values=context.problematic_values[:20],  # Keep more for analysis
            sample_size=len(context.sample_values),
            total_errors=len(context.problematic_values),
            error_percentage=(len(context.problematic_values) / max(1, len(context.sample_values))) * 100,
            suggested_actions=suggestions,
            context=error_details
        )
        
        self.error_reports.append(error_report)
        
        # Log to monitoring system with enhanced context
        try:
            log_data_quality_metric(
                f"validation_error_{rule.name}",
                len(context.sample_values),
                len(context.sample_values) - len(context.problematic_values),
                len(context.problematic_values),
                {
                    'rule_type': rule.rule_type.value,
                    'severity': rule.severity.value,
                    'error_type': type(error).__name__,
                    'execution_time_ms': context.execution_time * 1000,
                    'data_type': context.data_type,
                    'recovery_attempted': context.recovery_attempted
                }
            )
        except Exception as monitoring_error:
            self.logger.warning(f"Failed to log monitoring metric for error in {rule.name}: {monitoring_error}")
    
    def _generate_actionable_suggestions(self, rule: DataQualityRule, error: Exception, 
                                       context: ValidationExecutionContext) -> List[str]:
        """Generate actionable suggestions based on validation error context."""
        suggestions = []
        
        # Rule-specific suggestions
        if rule.rule_type == ValidationRule.NOT_NULL:
            suggestions.extend([
                "Check data source for missing value handling",
                "Consider using default values for missing data",
                "Verify data extraction process completeness"
            ])
        
        elif rule.rule_type == ValidationRule.RANGE:
            if 'min_value' in rule.parameters or 'max_value' in rule.parameters:
                min_val = rule.parameters.get('min_value', 'N/A')
                max_val = rule.parameters.get('max_value', 'N/A')
                suggestions.extend([
                    f"Verify data values are within expected range [{min_val}, {max_val}]",
                    "Check for data entry errors or system clock issues",
                    "Consider adjusting validation range based on business requirements"
                ])
        
        elif rule.rule_type == ValidationRule.FORMAT:
            suggestions.extend([
                "Verify data format consistency in source system",
                "Check for encoding issues or special characters",
                "Consider implementing data normalization before validation"
            ])
        
        elif rule.rule_type == ValidationRule.ENUM:
            valid_values = rule.parameters.get('valid_values', [])
            suggestions.extend([
                f"Ensure values are from allowed set: {valid_values[:5]}",
                "Check for case sensitivity or whitespace issues",
                "Consider adding data standardization step"
            ])
        
        elif rule.rule_type == ValidationRule.UNIQUE:
            suggestions.extend([
                "Check for duplicate data in source system",
                "Verify primary key constraints",
                "Consider deduplication process before validation"
            ])
        
        # Error-specific suggestions
        if "comparison" in str(error).lower() or "datetime" in str(error).lower():
            suggestions.extend([
                "Check for mixed data types in the field",
                "Implement data type conversion before validation",
                "Consider handling categorical data separately"
            ])
        
        if "parameter" in str(error).lower() or "argument" in str(error).lower():
            suggestions.extend([
                "Verify validation rule parameters are correct",
                "Check error handler method signatures",
                "Update validation rule configuration"
            ])
        
        # Data type specific suggestions
        if context.data_type in ['object', 'mixed']:
            suggestions.extend([
                "Implement data type detection and conversion",
                "Consider separate validation rules for different data types",
                "Add data cleaning step before validation"
            ])
        
        # Performance suggestions
        if context.execution_time > 1.0:  # More than 1 second
            suggestions.extend([
                "Consider optimizing validation logic for large datasets",
                "Implement sampling for performance-heavy validations",
                "Use vectorized operations where possible"
            ])
        
        return suggestions[:10]  # Limit to top 10 suggestions
    
    def _log_validation_metrics(self, validation_results: List[ValidationResult], 
                              rule_execution_stats: Dict[str, Any], duration: float) -> None:
        """Log comprehensive validation metrics and performance data."""
        # Calculate performance metrics
        total_records = sum(r.total_records for r in validation_results)
        total_valid = sum(r.valid_records for r in validation_results)
        total_invalid = sum(r.invalid_records for r in validation_results)
        
        # Rule performance analysis
        rule_performance = {}
        for result in validation_results:
            execution_time = result.details.get('execution_time_seconds', 0)
            rule_performance[result.rule_name] = execution_time
        
        # Error distribution
        error_distribution = {}
        for result in validation_results:
            if not result.passed:
                error_type = f"{result.rule_type.value}_{result.severity.value}"
                error_distribution[error_type] = error_distribution.get(error_type, 0) + 1
        
        # Calculate memory usage (approximate)
        import psutil
        memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Create comprehensive validation metrics
        self.validation_metrics = ValidationMetrics(
            total_rules_executed=rule_execution_stats['rules_executed_successfully'],
            successful_rules=sum(1 for r in validation_results if r.passed),
            failed_rules=sum(1 for r in validation_results if not r.passed),
            skipped_rules=rule_execution_stats['rules_skipped'],
            total_execution_time=duration,
            average_rule_time=duration / max(1, len(validation_results)),
            total_records_validated=total_records,
            total_valid_records=total_valid,
            total_invalid_records=total_invalid,
            overall_data_quality_score=((total_valid / max(1, total_records)) * 100),
            rule_performance=rule_performance,
            error_distribution=error_distribution,
            memory_usage_mb=memory_usage_mb
        )
        
        # Log comprehensive metrics summary
        self.logger.info(
            f"VALIDATION METRICS SUMMARY:\n"
            f"  Execution Performance:\n"
            f"    - Total Duration: {duration:.2f}s\n"
            f"    - Rules Executed: {rule_execution_stats['rules_executed_successfully']}\n"
            f"    - Rules Failed: {rule_execution_stats['rules_failed_execution']}\n"
            f"    - Rules Skipped: {rule_execution_stats['rules_skipped']}\n"
            f"    - Average Rule Time: {duration / max(1, len(validation_results)):.3f}s\n"
            f"    - Memory Usage: {memory_usage_mb:.2f}MB\n"
            f"  Data Quality Results:\n"
            f"    - Total Records: {total_records:,}\n"
            f"    - Valid Records: {total_valid:,} ({(total_valid/max(1,total_records)*100):.2f}%)\n"
            f"    - Invalid Records: {total_invalid:,} ({(total_invalid/max(1,total_records)*100):.2f}%)\n"
            f"    - Overall Quality Score: {(total_valid/max(1,total_records)*100):.2f}%"
        )
        
        # Log rule performance breakdown
        if rule_performance:
            sorted_performance = sorted(rule_performance.items(), key=lambda x: x[1], reverse=True)
            slowest_rules = sorted_performance[:5]  # Top 5 slowest rules
            
            self.logger.info(
                f"RULE PERFORMANCE BREAKDOWN (Top 5 Slowest):\n" +
                "\n".join([f"    - {rule}: {time:.3f}s" for rule, time in slowest_rules])
            )
        
        # Log error distribution
        if error_distribution:
            self.logger.warning(
                f"ERROR DISTRIBUTION:\n" +
                "\n".join([f"    - {error_type}: {count} occurrences" 
                          for error_type, count in sorted(error_distribution.items(), 
                                                         key=lambda x: x[1], reverse=True)])
            )
        
        # Log detailed performance metrics as JSON for monitoring systems
        metrics_json = {
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': duration,
            'rules_executed': rule_execution_stats['rules_executed_successfully'],
            'rules_failed': rule_execution_stats['rules_failed_execution'],
            'rules_skipped': rule_execution_stats['rules_skipped'],
            'total_records': total_records,
            'valid_records': total_valid,
            'invalid_records': total_invalid,
            'data_quality_score': (total_valid / max(1, total_records)) * 100,
            'memory_usage_mb': memory_usage_mb,
            'rule_performance': rule_performance,
            'error_distribution': error_distribution
        }
        
        self.logger.debug(f"Validation Metrics JSON: {json.dumps(metrics_json, indent=2)}")
        
        # Log actionable insights based on metrics
        self._log_actionable_insights(self.validation_metrics, rule_execution_stats, validation_results)
    
    def _log_problematic_values_analysis(self, validation_results: List[ValidationResult]) -> None:
        """Log detailed analysis of problematic values found during validation."""
        problematic_fields = {}
        
        for result in validation_results:
            if not result.passed and result.sample_invalid_values:
                field_name = result.field_name
                if field_name not in problematic_fields:
                    problematic_fields[field_name] = {
                        'rules_failed': [],
                        'sample_invalid_values': set(),
                        'total_invalid_count': 0,
                        'error_patterns': {}
                    }
                
                problematic_fields[field_name]['rules_failed'].append({
                    'rule_name': result.rule_name,
                    'rule_type': result.rule_type.value,
                    'severity': result.severity.value,
                    'error_rate': result.error_rate,
                    'invalid_count': result.invalid_records
                })
                
                # Collect sample invalid values (limit to avoid memory issues)
                for value in result.sample_invalid_values[:10]:
                    problematic_fields[field_name]['sample_invalid_values'].add(str(value))
                
                problematic_fields[field_name]['total_invalid_count'] += result.invalid_records
                
                # Analyze error patterns
                for value in result.sample_invalid_values[:20]:
                    value_str = str(value)
                    value_type = type(value).__name__
                    
                    # Pattern analysis
                    if value_str.lower() in ['null', 'none', 'nan', '']:
                        pattern = 'missing_values'
                    elif value_type in ['str', 'object'] and any(char.isdigit() for char in value_str):
                        pattern = 'mixed_alphanumeric'
                    elif value_type in ['str', 'object'] and len(value_str) > 100:
                        pattern = 'excessive_length'
                    elif value_type in ['str', 'object'] and not value_str.strip():
                        pattern = 'whitespace_only'
                    elif value_type in ['float', 'int'] and (value < 0 if isinstance(value, (int, float)) else False):
                        pattern = 'negative_values'
                    else:
                        pattern = f'type_{value_type}'
                    
                    problematic_fields[field_name]['error_patterns'][pattern] = \
                        problematic_fields[field_name]['error_patterns'].get(pattern, 0) + 1
        
        # Log problematic values analysis
        if problematic_fields:
            self.logger.warning("PROBLEMATIC VALUES ANALYSIS:")
            
            for field_name, field_data in problematic_fields.items():
                self.logger.warning(
                    f"  Field: {field_name}\n"
                    f"    Total Invalid Records: {field_data['total_invalid_count']:,}\n"
                    f"    Failed Rules: {len(field_data['rules_failed'])}\n"
                    f"    Sample Invalid Values: {list(field_data['sample_invalid_values'])[:5]}\n"
                    f"    Error Patterns: {dict(sorted(field_data['error_patterns'].items(), key=lambda x: x[1], reverse=True))}"
                )
                
                # Log specific rule failures for this field
                for rule_failure in field_data['rules_failed']:
                    self.logger.warning(
                        f"      Rule '{rule_failure['rule_name']}' ({rule_failure['rule_type']}): "
                        f"{rule_failure['invalid_count']:,} failures ({rule_failure['error_rate']:.2f}% error rate) "
                        f"[{rule_failure['severity'].upper()}]"
                    )
                
                # Generate field-specific recommendations
                recommendations = self._generate_field_recommendations(field_name, field_data)
                if recommendations:
                    self.logger.info(f"    Recommendations for {field_name}:")
                    for rec in recommendations:
                        self.logger.info(f"      - {rec}")
        
        # Log summary of most problematic fields
        if problematic_fields:
            sorted_fields = sorted(problematic_fields.items(), 
                                 key=lambda x: x[1]['total_invalid_count'], reverse=True)
            top_problematic = sorted_fields[:3]
            
            self.logger.error(
                f"TOP 3 MOST PROBLEMATIC FIELDS:\n" +
                "\n".join([f"    {i+1}. {field}: {data['total_invalid_count']:,} invalid records" 
                          for i, (field, data) in enumerate(top_problematic)])
            )
    
    def _generate_field_recommendations(self, field_name: str, field_data: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for problematic fields."""
        recommendations = []
        error_patterns = field_data['error_patterns']
        
        # Pattern-based recommendations
        if 'missing_values' in error_patterns:
            recommendations.append("Implement missing value handling (default values, imputation, or exclusion)")
        
        if 'mixed_alphanumeric' in error_patterns:
            recommendations.append("Add data type standardization and validation at source")
        
        if 'excessive_length' in error_patterns:
            recommendations.append("Implement string length validation and truncation rules")
        
        if 'whitespace_only' in error_patterns:
            recommendations.append("Add whitespace trimming and empty string handling")
        
        if 'negative_values' in error_patterns and field_name in ['quantity', 'unit_price', 'discount_percent']:
            recommendations.append("Add business rule validation for positive values")
        
        if 'type_object' in error_patterns and field_name in ['sale_date']:
            recommendations.append("Implement robust date parsing with multiple format support")
        
        # Field-specific recommendations
        if field_name == 'customer_email':
            recommendations.append("Add email format validation and normalization")
        elif field_name == 'product_name':
            recommendations.append("Implement product name standardization and deduplication")
        elif field_name in ['category', 'region']:
            recommendations.append("Create reference data validation with fuzzy matching")
        
        # Rule failure based recommendations
        failed_rule_types = [rule['rule_type'] for rule in field_data['rules_failed']]
        if 'unique' in failed_rule_types:
            recommendations.append("Investigate duplicate data sources and implement deduplication")
        
        if 'range' in failed_rule_types:
            recommendations.append("Review and adjust validation ranges based on business requirements")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _log_actionable_insights(self, metrics: ValidationMetrics, 
                               rule_execution_stats: Dict[str, Any],
                               validation_results: List[ValidationResult]) -> None:
        """Log actionable insights based on validation metrics."""
        insights = []
        
        # Performance insights
        if metrics.average_rule_time > 0.5:  # More than 500ms per rule
            insights.append(
                f"PERFORMANCE: Average rule execution time ({metrics.average_rule_time:.3f}s) is high. "
                f"Consider optimizing slow rules or implementing parallel validation."
            )
        
        if metrics.memory_usage_mb > 1000:  # More than 1GB
            insights.append(
                f"MEMORY: High memory usage ({metrics.memory_usage_mb:.0f}MB). "
                f"Consider implementing chunked validation or memory optimization."
            )
        
        # Data quality insights
        if metrics.overall_data_quality_score < 95:
            insights.append(
                f"DATA QUALITY: Overall quality score ({metrics.overall_data_quality_score:.1f}%) is below target. "
                f"Focus on top failing rules and implement data cleaning processes."
            )
        
        if metrics.failed_rules > metrics.successful_rules * 0.2:  # More than 20% failure rate
            insights.append(
                f"RULE RELIABILITY: High rule failure rate ({metrics.failed_rules}/{metrics.total_rules_executed}). "
                f"Review rule implementations and error handling."
            )
        
        # Execution insights
        if rule_execution_stats['rules_failed_execution'] > 0:
            insights.append(
                f"EXECUTION STABILITY: {rule_execution_stats['rules_failed_execution']} rules failed to execute. "
                f"Check rule implementations and data compatibility."
            )
        
        if rule_execution_stats['rules_skipped'] > rule_execution_stats['total_rules_attempted'] * 0.1:
            insights.append(
                f"RULE COVERAGE: {rule_execution_stats['rules_skipped']} rules were skipped. "
                f"Verify data schema compatibility and rule configurations."
            )
        
        # Error pattern insights
        if metrics.error_distribution:
            most_common_error = max(metrics.error_distribution.items(), key=lambda x: x[1])
            if most_common_error[1] > 5:  # More than 5 occurrences
                insights.append(
                    f"ERROR PATTERN: '{most_common_error[0]}' is the most common error type "
                    f"({most_common_error[1]} occurrences). Prioritize fixing this pattern."
                )
        
        # Log insights
        if insights:
            self.logger.info("ACTIONABLE INSIGHTS:")
            for i, insight in enumerate(insights, 1):
                self.logger.info(f"  {i}. {insight}")
        else:
            self.logger.info("VALIDATION HEALTH: All metrics are within acceptable ranges.")
    
    def _log_problematic_values_analysis(self, validation_results: List[ValidationResult]) -> None:
        """Log analysis of problematic values with patterns and insights."""
        if not validation_results:
            return
        
        # Analyze problematic values across all failed validations
        problematic_analysis = {}
        
        for result in validation_results:
            if not result.passed and result.sample_invalid_values:
                field_name = result.field_name
                rule_type = result.rule_type.value
                
                if field_name not in problematic_analysis:
                    problematic_analysis[field_name] = {
                        'total_invalid': 0,
                        'rules_failed': [],
                        'sample_values': set(),
                        'patterns': {}
                    }
                
                analysis = problematic_analysis[field_name]
                analysis['total_invalid'] += result.invalid_records
                analysis['rules_failed'].append(rule_type)
                analysis['sample_values'].update(result.sample_invalid_values[:5])
                
                # Pattern analysis
                for value in result.sample_invalid_values[:10]:
                    value_type = type(value).__name__
                    if value_type not in analysis['patterns']:
                        analysis['patterns'][value_type] = 0
                    analysis['patterns'][value_type] += 1
        
        # Log analysis results
        if problematic_analysis:
            self.logger.warning("PROBLEMATIC VALUES ANALYSIS:")
            
            for field_name, analysis in problematic_analysis.items():
                self.logger.warning(
                    f"  Field '{field_name}':\n"
                    f"    Total Invalid Records: {analysis['total_invalid']}\n"
                    f"    Failed Rules: {', '.join(analysis['rules_failed'])}\n"
                    f"    Sample Problematic Values: {list(analysis['sample_values'])[:5]}\n"
                    f"    Data Type Patterns: {dict(analysis['patterns'])}"
                )
                
                # Generate field-specific insights
                insights = self._generate_field_insights(field_name, analysis)
                if insights:
                    self.logger.info(f"    Insights: {'; '.join(insights)}")
    
    def _generate_field_insights(self, field_name: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate insights about problematic values for a specific field."""
        insights = []
        
        # Type diversity insight
        type_count = len(analysis['patterns'])
        if type_count > 2:
            insights.append(f"Mixed data types detected ({type_count} different types)")
        
        # Common patterns
        if 'NoneType' in analysis['patterns']:
            null_ratio = analysis['patterns']['NoneType'] / sum(analysis['patterns'].values())
            if null_ratio > 0.5:
                insights.append(f"High null value ratio ({null_ratio:.1%})")
        
        if 'str' in analysis['patterns'] and field_name in ['quantity', 'unit_price', 'discount_percent']:
            insights.append("Numeric field contains string values - check data source")
        
        # Rule-specific insights
        failed_rules = analysis['rules_failed']
        if 'range' in failed_rules and 'not_null' in failed_rules:
            insights.append("Both range and null validation failed - check data completeness")
        
        if 'format' in failed_rules:
            insights.append("Format validation failed - consider data standardization")
        
        if 'enum' in failed_rules:
            insights.append("Invalid category values - update allowed values or clean data")
        
        return insights
    
    def get_validation_error_report(self) -> Dict[str, Any]:
        """Get comprehensive validation error report with actionable insights."""
        if not self.error_reports:
            return {'status': 'no_errors', 'message': 'No validation errors recorded'}
        
        # Aggregate error statistics
        total_errors = len(self.error_reports)
        error_by_field = {}
        error_by_type = {}
        error_by_severity = {}
        
        for report in self.error_reports:
            # By field
            field = report.field_name
            if field not in error_by_field:
                error_by_field[field] = []
            error_by_field[field].append(report)
            
            # By error type
            error_type = report.error_type
            error_by_type[error_type] = error_by_type.get(error_type, 0) + 1
            
            # By severity (extract from context)
            severity = report.context.get('severity', 'unknown')
            error_by_severity[severity] = error_by_severity.get(severity, 0) + 1
        
        # Generate summary
        summary = {
            'total_errors': total_errors,
            'errors_by_field': {field: len(reports) for field, reports in error_by_field.items()},
            'errors_by_type': error_by_type,
            'errors_by_severity': error_by_severity,
            'most_problematic_fields': sorted(error_by_field.items(), key=lambda x: len(x[1]), reverse=True)[:5],
            'recent_errors': [
                {
                    'rule_name': report.rule_name,
                    'field_name': report.field_name,
                    'error_type': report.error_type,
                    'error_percentage': report.error_percentage,
                    'suggested_actions': report.suggested_actions[:3],
                    'timestamp': report.timestamp.isoformat()
                }
                for report in sorted(self.error_reports, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }
        
        return summary
    
    def export_validation_report(self, filepath: str) -> None:
        """Export comprehensive validation report to file."""
        report_data = {
            'validation_session': {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.validation_metrics.__dict__ if self.validation_metrics else None,
                'error_summary': self.get_validation_error_report()
            },
            'detailed_errors': [
                {
                    'rule_name': report.rule_name,
                    'field_name': report.field_name,
                    'error_type': report.error_type,
                    'error_message': report.error_message,
                    'problematic_values': report.problematic_values,
                    'error_percentage': report.error_percentage,
                    'suggested_actions': report.suggested_actions,
                    'context': report.context,
                    'timestamp': report.timestamp.isoformat()
                }
                for report in self.error_reports
            ],
            'validation_history': [
                {
                    'rule_name': result.rule_name,
                    'field_name': result.field_name,
                    'rule_type': result.rule_type.value,
                    'severity': result.severity.value,
                    'passed': result.passed,
                    'error_rate': result.error_rate,
                    'total_records': result.total_records,
                    'invalid_records': result.invalid_records,
                    'sample_invalid_values': result.sample_invalid_values,
                    'timestamp': result.timestamp.isoformat()
                }
                for result in self.validation_history[-50:]  # Last 50 results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Validation report exported to {filepath}")
    
    def _run_validation_rule(self, df: pd.DataFrame, rule: DataQualityRule) -> ValidationResult:
        """Run a single validation rule."""
        field_data = df[rule.field_name]
        total_records = len(field_data)
        
        # Run the validation function
        is_valid, details = rule.validation_function(field_data, **rule.parameters)
        
        # Calculate statistics
        valid_records = is_valid.sum()
        invalid_records = total_records - valid_records
        error_rate = (invalid_records / total_records) * 100 if total_records > 0 else 0
        
        # Determine if rule passed based on threshold
        passed = error_rate <= rule.threshold
        
        # Get sample invalid values
        sample_invalid_values = []
        if invalid_records > 0:
            invalid_mask = ~is_valid
            invalid_values = field_data[invalid_mask].dropna().unique()
            sample_invalid_values = list(invalid_values[:10])  # First 10 unique invalid values
        
        return ValidationResult(
            rule_name=rule.name,
            field_name=rule.field_name,
            rule_type=rule.rule_type,
            severity=rule.severity,
            passed=passed,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            error_rate=error_rate,
            details=details,
            sample_invalid_values=sample_invalid_values
        )
    
    def _run_validation_rule_with_isolation(self, df: pd.DataFrame, rule: DataQualityRule) -> Optional[ValidationResult]:
        """Run a single validation rule with comprehensive error isolation and recovery."""
        rule_start_time = time.time()
        
        # Create execution context for enhanced logging
        field_data = df[rule.field_name] if rule.field_name in df.columns else pd.Series(dtype=object)
        context = ValidationExecutionContext(
            rule_name=rule.name,
            field_name=rule.field_name,
            data_type=str(field_data.dtype),
            sample_values=list(field_data.dropna().head(10)) if not field_data.empty else [],
            error_category=ErrorCategory.VALIDATION,
            validation_parameters=rule.parameters.copy()
        )
        
        try:
            # Pre-validation checks with isolation
            total_records = len(field_data)
            
            if total_records == 0:
                self.logger.warning(f"Rule {rule.name}: Field {rule.field_name} has no data")
                return None
            
            self.logger.debug(f"Rule {rule.name}: Processing {total_records} records for field {rule.field_name} (type: {context.data_type})")
            
            # Run the validation function with isolation
            validation_function_failed = False
            try:
                is_valid, details = rule.validation_function(field_data, **rule.parameters)
                
                # Validate the validation function output
                if not isinstance(is_valid, pd.Series):
                    raise ValueError(f"Validation function returned invalid type: {type(is_valid)}")
                
                if len(is_valid) != total_records:
                    raise ValueError(f"Validation function returned wrong length: {len(is_valid)} vs {total_records}")
                
            except Exception as validation_error:
                # Validation function failed - create fallback result
                validation_function_failed = True
                context.fallback_applied = True
                context.error_details = {
                    'validation_error': str(validation_error),
                    'validation_error_type': type(validation_error).__name__
                }
                
                # Enhanced error logging with context
                self._log_validation_error_with_context(rule, validation_error, context)
                
                # Record this as an execution error for tracking
                if hasattr(self, '_current_execution_stats'):
                    self._current_execution_stats['execution_errors'].append({
                        'rule_name': rule.name,
                        'field_name': rule.field_name,
                        'error_type': 'validation_function_failure',
                        'error_message': str(validation_error),
                        'severity': 'error',
                        'exception_type': type(validation_error).__name__
                    })
                
                # Create a fallback result where all records are considered invalid
                is_valid = pd.Series(False, index=field_data.index)
                details = {
                    'validation_error': str(validation_error),
                    'validation_error_type': type(validation_error).__name__,
                    'fallback_applied': True,
                    'validation_type': f'{rule.rule_type.value}_failed'
                }
                
                # Log the fallback application
                self.logger.warning(f"Rule {rule.name}: Applied fallback - marking all records as invalid due to validation function failure")
            
            # Calculate statistics with error handling
            try:
                valid_records = int(is_valid.sum())
                invalid_records = total_records - valid_records
                error_rate = (invalid_records / total_records) * 100 if total_records > 0 else 0
                
                # Determine if rule passed based on threshold
                passed = error_rate <= rule.threshold
                
            except Exception as stats_error:
                self.logger.error(f"Rule {rule.name}: Statistics calculation failed: {stats_error}")
                # Fallback statistics
                valid_records = 0
                invalid_records = total_records
                error_rate = 100.0
                passed = False
                
                details['statistics_error'] = str(stats_error)
                details['statistics_fallback_applied'] = True
                
                # Update context with statistics error
                context.error_details['statistics_error'] = str(stats_error)
            
            # Get sample invalid values with enhanced error handling and context
            sample_invalid_values = []
            try:
                if invalid_records > 0:
                    invalid_mask = ~is_valid
                    invalid_values = field_data[invalid_mask].dropna().unique()
                    sample_invalid_values = list(invalid_values[:10])  # First 10 unique invalid values
                    
                    # Update context with problematic values
                    context.problematic_values = list(invalid_values[:20])  # Keep more for analysis
                    
            except Exception as sample_error:
                self.logger.warning(f"Rule {rule.name}: Failed to extract sample invalid values: {sample_error}")
                sample_invalid_values = []
                details['sample_extraction_error'] = str(sample_error)
                context.error_details['sample_extraction_error'] = str(sample_error)
            
            # Add execution timing and metadata
            rule_duration = time.time() - rule_start_time
            context.execution_time = rule_duration
            details['execution_time_seconds'] = rule_duration
            details['execution_timestamp'] = datetime.now().isoformat()
            
            # Enhanced logging for validation results
            if not passed:
                self.logger.warning(
                    f"Rule {rule.name} FAILED: {invalid_records}/{total_records} invalid records "
                    f"({error_rate:.2f}% error rate, threshold: {rule.threshold}%)\n"
                    f"  Sample Invalid Values: {sample_invalid_values[:5]}\n"
                    f"  Execution Time: {rule_duration*1000:.2f}ms"
                )
                
                # Log problematic values analysis if significant errors
                if error_rate > 10:  # More than 10% error rate
                    temp_result = ValidationResult(
                        rule_name=rule.name,
                        field_name=rule.field_name,
                        rule_type=rule.rule_type,
                        severity=rule.severity,
                        passed=passed,
                        total_records=total_records,
                        valid_records=valid_records,
                        invalid_records=invalid_records,
                        error_rate=error_rate,
                        details=details,
                        sample_invalid_values=sample_invalid_values
                    )
                    self._log_problematic_values_analysis([temp_result])
            else:
                self.logger.debug(
                    f"Rule {rule.name} PASSED: {valid_records}/{total_records} valid records "
                    f"({error_rate:.2f}% error rate) in {rule_duration*1000:.2f}ms"
                )
            
            # Create validation result
            result = ValidationResult(
                rule_name=rule.name,
                field_name=rule.field_name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                passed=passed,
                total_records=total_records,
                valid_records=valid_records,
                invalid_records=invalid_records,
                error_rate=error_rate,
                details=details,
                sample_invalid_values=sample_invalid_values
            )
            
            return result
            
        except Exception as critical_error:
            # Critical error in rule execution - log with enhanced context
            rule_duration = time.time() - rule_start_time
            context.execution_time = rule_duration
            context.error_details['critical_error'] = str(critical_error)
            context.error_details['critical_error_type'] = type(critical_error).__name__
            
            # Enhanced error logging with full context
            self._log_validation_error_with_context(rule, critical_error, context)
            
            error_context = {
                'rule_name': rule.name,
                'field_name': rule.field_name,
                'rule_type': rule.rule_type.value,
                'severity': rule.severity.value,
                'execution_time': rule_duration,
                'error_type': type(critical_error).__name__,
                'error_message': str(critical_error),
                'data_type': context.data_type,
                'sample_values': context.sample_values[:5]
            }
            
            # Try to handle the error through the error handler, but don't let it crash validation
            try:
                error_handler.handle_error(
                    DataQualityError(f"Critical validation rule failure: {rule.name}", rule.field_name),
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.HIGH,
                    context=error_context
                )
            except Exception as handler_error:
                self.logger.error(f"Rule {rule.name}: Error handler also failed: {handler_error}")
            
            return None
    
    def create_structured_error_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Create a comprehensive structured error report for debugging and analysis."""
        report_timestamp = datetime.now()
        
        # Aggregate error data from validation history and error reports
        error_summary = {
            'report_metadata': {
                'generated_at': report_timestamp.isoformat(),
                'total_validation_sessions': len(self.validation_history),
                'total_error_reports': len(self.error_reports),
                'report_version': '1.0'
            },
            'validation_overview': {
                'total_rules_run': len(self.validation_history),
                'successful_validations': sum(1 for r in self.validation_history if r.passed),
                'failed_validations': sum(1 for r in self.validation_history if not r.passed),
                'average_error_rate': sum(r.error_rate for r in self.validation_history) / max(1, len(self.validation_history))
            },
            'error_analysis': {
                'by_field': {},
                'by_rule_type': {},
                'by_severity': {},
                'common_patterns': {},
                'problematic_values': {}
            },
            'performance_analysis': {
                'slowest_rules': [],
                'memory_usage_trends': [],
                'execution_failures': []
            },
            'actionable_recommendations': []
        }
        
        # Analyze errors by field
        field_errors = {}
        for result in self.validation_history:
            if not result.passed:
                field = result.field_name
                if field not in field_errors:
                    field_errors[field] = {
                        'total_failures': 0,
                        'total_invalid_records': 0,
                        'failed_rules': [],
                        'sample_invalid_values': set(),
                        'error_patterns': {}
                    }
                
                field_errors[field]['total_failures'] += 1
                field_errors[field]['total_invalid_records'] += result.invalid_records
                field_errors[field]['failed_rules'].append({
                    'rule_name': result.rule_name,
                    'rule_type': result.rule_type.value,
                    'severity': result.severity.value,
                    'error_rate': result.error_rate,
                    'timestamp': result.timestamp.isoformat()
                })
                
                # Collect sample values
                for value in result.sample_invalid_values[:5]:
                    field_errors[field]['sample_invalid_values'].add(str(value))
        
        error_summary['error_analysis']['by_field'] = {
            field: {
                'total_failures': data['total_failures'],
                'total_invalid_records': data['total_invalid_records'],
                'failed_rules_count': len(data['failed_rules']),
                'sample_invalid_values': list(data['sample_invalid_values']),
                'most_recent_failure': max(data['failed_rules'], key=lambda x: x['timestamp'])['timestamp'] if data['failed_rules'] else None
            }
            for field, data in field_errors.items()
        }
        
        # Analyze errors by rule type
        rule_type_errors = {}
        for result in self.validation_history:
            if not result.passed:
                rule_type = result.rule_type.value
                if rule_type not in rule_type_errors:
                    rule_type_errors[rule_type] = {
                        'failure_count': 0,
                        'total_invalid_records': 0,
                        'affected_fields': set()
                    }
                
                rule_type_errors[rule_type]['failure_count'] += 1
                rule_type_errors[rule_type]['total_invalid_records'] += result.invalid_records
                rule_type_errors[rule_type]['affected_fields'].add(result.field_name)
        
        error_summary['error_analysis']['by_rule_type'] = {
            rule_type: {
                'failure_count': data['failure_count'],
                'total_invalid_records': data['total_invalid_records'],
                'affected_fields': list(data['affected_fields'])
            }
            for rule_type, data in rule_type_errors.items()
        }
        
        # Analyze errors by severity
        severity_errors = {}
        for result in self.validation_history:
            if not result.passed:
                severity = result.severity.value
                severity_errors[severity] = severity_errors.get(severity, 0) + 1
        
        error_summary['error_analysis']['by_severity'] = severity_errors
        
        # Performance analysis
        if self.validation_history:
            execution_times = [(r.rule_name, r.details.get('execution_time_seconds', 0)) 
                             for r in self.validation_history]
            slowest_rules = sorted(execution_times, key=lambda x: x[1], reverse=True)[:10]
            
            error_summary['performance_analysis']['slowest_rules'] = [
                {'rule_name': rule, 'execution_time_seconds': time}
                for rule, time in slowest_rules
            ]
        
        # Generate actionable recommendations
        recommendations = self._generate_comprehensive_recommendations(field_errors, rule_type_errors, severity_errors)
        error_summary['actionable_recommendations'] = recommendations
        
        # Save report to file if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    json.dump(error_summary, f, indent=2, default=str)
                self.logger.info(f"Structured error report saved to: {output_path}")
            except Exception as save_error:
                self.logger.error(f"Failed to save error report: {save_error}")
        
        return error_summary
    
    def _generate_comprehensive_recommendations(self, field_errors: Dict[str, Any], 
                                             rule_type_errors: Dict[str, Any],
                                             severity_errors: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive actionable recommendations based on error analysis."""
        recommendations = []
        
        # Field-specific recommendations
        if field_errors:
            most_problematic_field = max(field_errors.items(), key=lambda x: x[1]['total_invalid_records'])
            field_name, field_data = most_problematic_field
            
            recommendations.append({
                'category': 'data_quality',
                'priority': 'high',
                'title': f'Address data quality issues in field: {field_name}',
                'description': f'Field {field_name} has {field_data["total_invalid_records"]} invalid records across {field_data["total_failures"]} validation failures.',
                'suggested_actions': [
                    f'Review data source quality for {field_name}',
                    f'Implement data cleaning rules for {field_name}',
                    f'Add data validation at ingestion point',
                    f'Consider default value strategies for {field_name}'
                ],
                'impact': 'Reducing errors in this field will significantly improve overall data quality'
            })
        
        # Rule type recommendations
        if rule_type_errors:
            most_failing_rule_type = max(rule_type_errors.items(), key=lambda x: x[1]['failure_count'])
            rule_type, rule_data = most_failing_rule_type
            
            recommendations.append({
                'category': 'validation_rules',
                'priority': 'medium',
                'title': f'Optimize {rule_type} validation rules',
                'description': f'{rule_type} rules are failing frequently ({rule_data["failure_count"]} failures) across {len(rule_data["affected_fields"])} fields.',
                'suggested_actions': [
                    f'Review {rule_type} rule implementations',
                    f'Adjust {rule_type} rule thresholds if appropriate',
                    f'Add error handling for {rule_type} rules',
                    f'Consider rule-specific data preprocessing'
                ],
                'impact': f'Improving {rule_type} rules will reduce validation failures across multiple fields'
            })
        
        # Severity-based recommendations
        if 'critical' in severity_errors and severity_errors['critical'] > 0:
            recommendations.append({
                'category': 'system_stability',
                'priority': 'critical',
                'title': 'Address critical validation failures',
                'description': f'{severity_errors["critical"]} critical validation failures detected.',
                'suggested_actions': [
                    'Investigate critical validation failures immediately',
                    'Implement emergency data quality checks',
                    'Add monitoring alerts for critical failures',
                    'Review data pipeline stability'
                ],
                'impact': 'Critical failures can cause pipeline instability and data corruption'
            })
        
        # Performance recommendations
        if hasattr(self, 'validation_metrics') and self.validation_metrics:
            if self.validation_metrics.average_rule_time > 0.5:
                recommendations.append({
                    'category': 'performance',
                    'priority': 'medium',
                    'title': 'Optimize validation performance',
                    'description': f'Average rule execution time is {self.validation_metrics.average_rule_time:.3f}s, which may impact pipeline performance.',
                    'suggested_actions': [
                        'Profile slow validation rules',
                        'Implement parallel validation where possible',
                        'Consider sampling for expensive validations',
                        'Optimize data access patterns'
                    ],
                    'impact': 'Performance optimization will reduce overall pipeline execution time'
                })
        
        return recommendations
    
    def export_debugging_report(self, output_dir: str = "logs") -> str:
        """Export a comprehensive debugging report with all validation details."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"validation_debug_report_{timestamp}.json"
        report_path = Path(output_dir) / report_filename
        
        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create comprehensive debugging report
        debug_report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'validation_debugging',
                'version': '1.0'
            },
            'validation_history': [
                {
                    'rule_name': r.rule_name,
                    'field_name': r.field_name,
                    'rule_type': r.rule_type.value,
                    'severity': r.severity.value,
                    'passed': r.passed,
                    'total_records': r.total_records,
                    'valid_records': r.valid_records,
                    'invalid_records': r.invalid_records,
                    'error_rate': r.error_rate,
                    'sample_invalid_values': r.sample_invalid_values,
                    'details': r.details,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.validation_history[-100:]  # Last 100 validation results
            ],
            'error_reports': [
                {
                    'rule_name': er.rule_name,
                    'field_name': er.field_name,
                    'error_type': er.error_type,
                    'error_message': er.error_message,
                    'problematic_values': er.problematic_values,
                    'total_errors': er.total_errors,
                    'error_percentage': er.error_percentage,
                    'suggested_actions': er.suggested_actions,
                    'context': er.context,
                    'timestamp': er.timestamp.isoformat()
                }
                for er in self.error_reports[-50:]  # Last 50 error reports
            ],
            'validation_metrics': {
                'total_rules_executed': self.validation_metrics.total_rules_executed if self.validation_metrics else 0,
                'successful_rules': self.validation_metrics.successful_rules if self.validation_metrics else 0,
                'failed_rules': self.validation_metrics.failed_rules if self.validation_metrics else 0,
                'overall_data_quality_score': self.validation_metrics.overall_data_quality_score if self.validation_metrics else 0,
                'rule_performance': self.validation_metrics.rule_performance if self.validation_metrics else {},
                'error_distribution': self.validation_metrics.error_distribution if self.validation_metrics else {},
                'memory_usage_mb': self.validation_metrics.memory_usage_mb if self.validation_metrics else 0
            } if self.validation_metrics else None,
            'registered_rules': [
                {
                    'name': rule.name,
                    'field_name': rule.field_name,
                    'rule_type': rule.rule_type.value,
                    'severity': rule.severity.value,
                    'enabled': rule.enabled,
                    'threshold': rule.threshold,
                    'parameters': rule.parameters,
                    'description': rule.description
                }
                for rule in self.validation_rules.values()
            ]
        }
        
        # Save the report
        try:
            with open(report_path, 'w') as f:
                json.dump(debug_report, f, indent=2, default=str)
            
            self.logger.info(f"Debugging report exported to: {report_path}")
            return str(report_path)
            
        except Exception as export_error:
            self.logger.error(f"Failed to export debugging report: {export_error}")
            return ""
    
    def _generate_comprehensive_validation_summary(self, validation_results: List[ValidationResult],
                                                 rule_execution_stats: Dict[str, Any],
                                                 overall_status: str, critical_failures: int,
                                                 error_failures: int, warning_failures: int,
                                                 total_records: int, duration: float) -> Dict[str, Any]:
        """Generate comprehensive validation summary with success/failure metrics."""
        
        # Calculate comprehensive metrics
        total_rules_attempted = rule_execution_stats['total_rules_attempted']
        rules_executed_successfully = rule_execution_stats['rules_executed_successfully']
        rules_failed_execution = rule_execution_stats['rules_failed_execution']
        rules_skipped = rule_execution_stats['rules_skipped']
        
        # Rule execution success rate
        rule_execution_success_rate = (rules_executed_successfully / max(1, total_rules_attempted)) * 100
        
        # Validation success metrics (for successfully executed rules)
        validation_success_metrics = {
            'total_validations': len(validation_results),
            'passed_validations': sum(1 for r in validation_results if r.passed),
            'failed_validations': sum(1 for r in validation_results if not r.passed),
            'validation_success_rate': 0.0
        }
        
        if validation_success_metrics['total_validations'] > 0:
            validation_success_metrics['validation_success_rate'] = (
                validation_success_metrics['passed_validations'] / 
                validation_success_metrics['total_validations']
            ) * 100
        
        # Data quality metrics aggregation
        total_records_validated = sum(r.total_records for r in validation_results)
        total_valid_records = sum(r.valid_records for r in validation_results)
        total_invalid_records = sum(r.invalid_records for r in validation_results)
        
        overall_data_quality_rate = 0.0
        if total_records_validated > 0:
            overall_data_quality_rate = (total_valid_records / total_records_validated) * 100
        
        # Severity breakdown
        severity_breakdown = {
            'critical': {'count': critical_failures, 'rules': []},
            'error': {'count': error_failures, 'rules': []},
            'warning': {'count': warning_failures, 'rules': []},
            'info': {'count': 0, 'rules': []}
        }
        
        # Populate severity breakdown with rule details
        for result in validation_results:
            if not result.passed:
                severity_key = result.severity.value
                if severity_key in severity_breakdown:
                    severity_breakdown[severity_key]['rules'].append({
                        'rule_name': result.rule_name,
                        'field_name': result.field_name,
                        'error_rate': result.error_rate,
                        'invalid_records': result.invalid_records
                    })
        
        # Field-level analysis
        field_analysis = {}
        for result in validation_results:
            field_name = result.field_name
            if field_name not in field_analysis:
                field_analysis[field_name] = {
                    'total_rules': 0,
                    'passed_rules': 0,
                    'failed_rules': 0,
                    'worst_error_rate': 0.0,
                    'average_error_rate': 0.0,
                    'rule_details': []
                }
            
            field_stats = field_analysis[field_name]
            field_stats['total_rules'] += 1
            field_stats['rule_details'].append({
                'rule_name': result.rule_name,
                'rule_type': result.rule_type.value,
                'passed': result.passed,
                'error_rate': result.error_rate,
                'severity': result.severity.value
            })
            
            if result.passed:
                field_stats['passed_rules'] += 1
            else:
                field_stats['failed_rules'] += 1
            
            field_stats['worst_error_rate'] = max(field_stats['worst_error_rate'], result.error_rate)
        
        # Calculate average error rates for each field
        for field_name, field_stats in field_analysis.items():
            if field_stats['rule_details']:
                field_stats['average_error_rate'] = sum(
                    rule['error_rate'] for rule in field_stats['rule_details']
                ) / len(field_stats['rule_details'])
        
        # Performance metrics
        performance_metrics = {
            'total_duration_seconds': duration,
            'average_rule_duration': duration / max(1, rules_executed_successfully),
            'records_per_second': total_records / max(0.001, duration),
            'rules_per_second': rules_executed_successfully / max(0.001, duration)
        }
        
        # Generate comprehensive summary
        summary = {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            
            # Rule execution metrics
            'rule_execution_metrics': {
                'total_rules_attempted': total_rules_attempted,
                'rules_executed_successfully': rules_executed_successfully,
                'rules_failed_execution': rules_failed_execution,
                'rules_skipped': rules_skipped,
                'rule_execution_success_rate': rule_execution_success_rate,
                'execution_errors': rule_execution_stats['execution_errors']
            },
            
            # Validation success metrics
            'validation_success_metrics': validation_success_metrics,
            
            # Data quality metrics
            'data_quality_metrics': {
                'total_records_processed': total_records,
                'total_records_validated': total_records_validated,
                'total_valid_records': total_valid_records,
                'total_invalid_records': total_invalid_records,
                'overall_data_quality_rate': overall_data_quality_rate
            },
            
            # Failure breakdown
            'failure_breakdown': {
                'critical_failures': critical_failures,
                'error_failures': error_failures,
                'warning_failures': warning_failures,
                'severity_breakdown': severity_breakdown
            },
            
            # Field-level analysis
            'field_analysis': field_analysis,
            
            # Performance metrics
            'performance_metrics': performance_metrics,
            
            # Detailed results (for backward compatibility)
            'total_rules_run': len(validation_results),
            'total_records_validated': total_records,
            'validation_duration': duration,
            'results': [
                {
                    'rule_name': r.rule_name,
                    'field_name': r.field_name,
                    'rule_type': r.rule_type.value,
                    'severity': r.severity.value,
                    'passed': r.passed,
                    'error_rate': r.error_rate,
                    'invalid_records': r.invalid_records,
                    'valid_records': r.valid_records,
                    'total_records': r.total_records,
                    'details': r.details,
                    'sample_invalid_values': r.sample_invalid_values[:5]  # Limit sample size in summary
                }
                for r in validation_results
            ]
        }
        
        return summary
    
    # Validation function implementations
    def _validate_not_null(self, series: pd.Series, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Validate that values are not null."""
        is_valid = series.notna()
        null_count = series.isna().sum()
        
        details = {
            'null_count': int(null_count),
            'validation_type': 'not_null'
        }
        
        return is_valid, details
    
    def _validate_unique(self, series: pd.Series, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Validate that values are unique."""
        is_valid = ~series.duplicated()
        duplicate_count = series.duplicated().sum()
        
        details = {
            'duplicate_count': int(duplicate_count),
            'unique_count': int(series.nunique()),
            'validation_type': 'unique'
        }
        
        return is_valid, details
    
    def _validate_numeric_range(self, series: pd.Series, min_value: float = None, 
                              max_value: float = None, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Validate that numeric values are within specified range."""
        # Convert to numeric, invalid values become NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        is_valid = pd.Series(True, index=series.index)
        
        if min_value is not None:
            is_valid &= (numeric_series >= min_value) | numeric_series.isna()
        
        if max_value is not None:
            is_valid &= (numeric_series <= max_value) | numeric_series.isna()
        
        # Handle NaN values from conversion
        is_valid &= numeric_series.notna()
        
        out_of_range_count = (~is_valid).sum()
        
        details = {
            'min_value': min_value,
            'max_value': max_value,
            'out_of_range_count': int(out_of_range_count),
            'actual_min': float(numeric_series.min()) if not numeric_series.empty else None,
            'actual_max': float(numeric_series.max()) if not numeric_series.empty else None,
            'validation_type': 'numeric_range'
        }
        
        return is_valid, details
    
    def _validate_string_length(self, series: pd.Series, min_length: int = 0, 
                              max_length: int = None, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Validate string length."""
        string_series = series.astype(str)
        lengths = string_series.str.len()
        
        is_valid = lengths >= min_length
        if max_length is not None:
            is_valid &= lengths <= max_length
        
        invalid_count = (~is_valid).sum()
        
        details = {
            'min_length': min_length,
            'max_length': max_length,
            'invalid_length_count': int(invalid_count),
            'avg_length': float(lengths.mean()),
            'validation_type': 'string_length'
        }
        
        return is_valid, details
    
    def _validate_enum(self, series: pd.Series, valid_values: List[Any], **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Validate that values are in allowed set."""
        is_valid = series.isin(valid_values) | series.isna()
        invalid_count = (~is_valid).sum()
        
        # Get unique invalid values
        invalid_values = series[~is_valid].unique()
        
        details = {
            'valid_values': valid_values,
            'invalid_count': int(invalid_count),
            'invalid_values': list(invalid_values[:10]),  # First 10 invalid values
            'validation_type': 'enum'
        }
        
        return is_valid, details
    
    def _validate_email_format(self, series: pd.Series, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Validate email format."""
        import re
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        # Allow null values
        is_valid = series.isna() | series.str.match(email_pattern, na=False)
        invalid_count = (~is_valid).sum()
        
        details = {
            'invalid_format_count': int(invalid_count),
            'pattern': email_pattern,
            'validation_type': 'email_format'
        }
        
        return is_valid, details
    
    def _validate_date_range(self, series: pd.Series, min_date: str = None, 
                           max_date: str = None, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Validate date range with robust data type handling."""
        return self._validate_date_range_robust(series, min_date, max_date, **kwargs)
    
    def _validate_date_range_robust(self, series: pd.Series, min_date: str = None, 
                                  max_date: str = None, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Enhanced date validation method that detects data types first and handles categorical data gracefully."""
        
        # Initialize tracking variables
        total_records = len(series)
        conversion_stats = {
            'original_dtype': str(series.dtype),
            'null_count': int(series.isna().sum()),
            'categorical_count': 0,
            'numeric_count': 0,
            'string_count': 0,
            'datetime_count': 0,
            'conversion_attempts': 0,
            'conversion_successes': 0,
            'conversion_failures': 0,
            'fallback_applied': 0
        }
        
        # Step 1: Data type detection and categorization
        self.logger.debug(f"Starting robust date validation for {total_records} records")
        self.logger.debug(f"Original data type: {series.dtype}")
        
        # Handle empty series
        if series.empty:
            return pd.Series(dtype=bool), {
                'validation_type': 'date_range_robust',
                'conversion_stats': conversion_stats,
                'error_details': [],
                'warning_messages': ['Empty series provided for date validation']
            }
        
        # Detect data types in the series
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            # All values are null
            is_valid = pd.Series(True, index=series.index)  # Null values are considered valid
            conversion_stats['warning_messages'] = ['All values are null']
            return is_valid, {
                'validation_type': 'date_range_robust',
                'conversion_stats': conversion_stats,
                'error_details': [],
                'warning_messages': ['All values are null - validation skipped']
            }
        
        # Analyze data types present in the series
        for value in non_null_series.iloc[:min(100, len(non_null_series))]:  # Sample first 100 non-null values
            if pd.api.types.is_categorical_dtype(type(value)) or isinstance(value, str):
                if str(value).lower() in ['unknown', 'n/a', 'null', 'none', '']:
                    conversion_stats['categorical_count'] += 1
                else:
                    conversion_stats['string_count'] += 1
            elif pd.api.types.is_numeric_dtype(type(value)):
                conversion_stats['numeric_count'] += 1
            elif pd.api.types.is_datetime64_any_dtype(type(value)):
                conversion_stats['datetime_count'] += 1
            else:
                conversion_stats['string_count'] += 1
        
        # Step 2: Safe type conversion with fallback mechanisms
        error_details = []
        warning_messages = []
        
        try:
            # First attempt: Direct datetime conversion
            conversion_stats['conversion_attempts'] += 1
            date_series = pd.to_datetime(series, errors='coerce')
            
            # Check conversion success rate
            successful_conversions = date_series.notna().sum() - series.isna().sum()  # Exclude originally null values
            original_non_null = series.notna().sum()
            
            if original_non_null > 0:
                conversion_success_rate = successful_conversions / original_non_null
                conversion_stats['conversion_successes'] = int(successful_conversions)
                conversion_stats['conversion_failures'] = int(original_non_null - successful_conversions)
                
                if conversion_success_rate < 0.5:  # Less than 50% conversion success
                    warning_messages.append(f"Low conversion success rate: {conversion_success_rate:.2%}")
                    
                    # Step 3: Fallback mechanism for categorical/problematic data
                    conversion_stats['fallback_applied'] = 1
                    
                    # Try to handle common categorical date representations
                    fallback_series = series.copy()
                    
                    # Handle categorical data gracefully
                    if pd.api.types.is_categorical_dtype(series):
                        self.logger.debug("Detected categorical data type, applying categorical fallback")
                        # Convert categorical to string first, then to datetime
                        try:
                            string_series = fallback_series.astype(str)
                            # Replace common non-date categorical values
                            string_series = string_series.replace({
                                'Unknown': pd.NaT,
                                'unknown': pd.NaT, 
                                'N/A': pd.NaT,
                                'n/a': pd.NaT,
                                'NULL': pd.NaT,
                                'null': pd.NaT,
                                'None': pd.NaT,
                                'none': pd.NaT,
                                '': pd.NaT
                            })
                            
                            # Attempt conversion again
                            fallback_date_series = pd.to_datetime(string_series, errors='coerce')
                            
                            # Use fallback if it's better
                            fallback_success = fallback_date_series.notna().sum() - series.isna().sum()
                            if fallback_success > successful_conversions:
                                date_series = fallback_date_series
                                conversion_stats['conversion_successes'] = int(fallback_success)
                                conversion_stats['conversion_failures'] = int(original_non_null - fallback_success)
                                warning_messages.append("Applied categorical data fallback conversion")
                                
                        except Exception as fallback_error:
                            error_details.append({
                                'error_type': 'fallback_conversion_error',
                                'message': str(fallback_error),
                                'context': 'categorical_fallback'
                            })
                    
                    # Additional fallback: Try different date formats
                    if conversion_stats['conversion_failures'] > 0:
                        try:
                            # Try common date formats individually
                            common_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d', '%Y-%m-%d %H:%M:%S']
                            best_format_series = date_series
                            best_format_success = conversion_stats['conversion_successes']
                            best_format = None
                            
                            for fmt in common_formats:
                                try:
                                    format_series = pd.to_datetime(series, format=fmt, errors='coerce')
                                    format_success = format_series.notna().sum() - series.isna().sum()
                                    if format_success > best_format_success:
                                        best_format_series = format_series
                                        best_format_success = format_success
                                        best_format = fmt
                                except:
                                    continue
                            
                            # Apply the best format if it's better than the original
                            if best_format_success > conversion_stats['conversion_successes']:
                                date_series = best_format_series
                                conversion_stats['conversion_successes'] = int(best_format_success)
                                conversion_stats['conversion_failures'] = int(original_non_null - best_format_success)
                                warning_messages.append(f"Applied format-specific conversion: {best_format}")
                            
                            # If still have failures, try with dayfirst parameter variations
                            if conversion_stats['conversion_failures'] > 0:
                                try:
                                    # Try with dayfirst=True for European format dates
                                    dayfirst_series = pd.to_datetime(series, errors='coerce', dayfirst=True)
                                    dayfirst_success = dayfirst_series.notna().sum() - series.isna().sum()
                                    if dayfirst_success > conversion_stats['conversion_successes']:
                                        date_series = dayfirst_series
                                        conversion_stats['conversion_successes'] = int(dayfirst_success)
                                        conversion_stats['conversion_failures'] = int(original_non_null - dayfirst_success)
                                        warning_messages.append("Applied day-first datetime parsing")
                                except:
                                    pass
                                    
                        except Exception as format_error:
                            error_details.append({
                                'error_type': 'format_conversion_error', 
                                'message': str(format_error),
                                'context': 'format_fallback'
                            })
            
        except Exception as conversion_error:
            # Complete fallback: treat all as invalid but don't crash
            error_details.append({
                'error_type': 'primary_conversion_error',
                'message': str(conversion_error),
                'context': 'primary_conversion'
            })
            
            # Create a series where all non-null values are considered invalid for date validation
            date_series = pd.Series(pd.NaT, index=series.index)
            conversion_stats['conversion_failures'] = int(series.notna().sum())
            warning_messages.append("Primary conversion failed, treating all non-null values as invalid dates")
        
        # Step 4: Apply date range validation
        is_valid = pd.Series(True, index=series.index)
        
        # Null values are considered valid (they pass validation)
        is_valid = series.isna() | date_series.notna()
        
        # Apply min_date constraint
        if min_date and date_series.notna().any():
            try:
                min_dt = pd.to_datetime(min_date)
                date_constraint = (date_series >= min_dt) | date_series.isna()
                is_valid &= date_constraint
            except Exception as min_date_error:
                error_details.append({
                    'error_type': 'min_date_parsing_error',
                    'message': str(min_date_error),
                    'context': f'min_date: {min_date}'
                })
        
        # Apply max_date constraint  
        if max_date and date_series.notna().any():
            try:
                max_dt = pd.to_datetime(max_date)
                date_constraint = (date_series <= max_dt) | date_series.isna()
                is_valid &= date_constraint
            except Exception as max_date_error:
                error_details.append({
                    'error_type': 'max_date_parsing_error',
                    'message': str(max_date_error),
                    'context': f'max_date: {max_date}'
                })
        
        # Step 5: Comprehensive error reporting
        out_of_range_count = (~is_valid).sum()
        
        # Get sample problematic values for debugging
        sample_invalid_values = []
        sample_unconvertible_values = []
        
        if out_of_range_count > 0:
            invalid_mask = ~is_valid
            sample_invalid_values = list(series[invalid_mask].dropna().unique()[:5])
        
        if conversion_stats['conversion_failures'] > 0:
            # Find values that couldn't be converted to dates
            conversion_failed_mask = series.notna() & date_series.isna()
            sample_unconvertible_values = list(series[conversion_failed_mask].unique()[:5])
        
        # Calculate actual date range from successfully converted dates
        actual_min_date = None
        actual_max_date = None
        if date_series.notna().any():
            try:
                actual_min_date = str(date_series.min())
                actual_max_date = str(date_series.max())
            except:
                pass
        
        # Detailed validation results
        details = {
            'min_date': min_date,
            'max_date': max_date,
            'out_of_range_count': int(out_of_range_count),
            'actual_min_date': actual_min_date,
            'actual_max_date': actual_max_date,
            'validation_type': 'date_range_robust',
            'conversion_stats': conversion_stats,
            'error_details': error_details,
            'warning_messages': warning_messages,
            'sample_invalid_values': sample_invalid_values,
            'sample_unconvertible_values': sample_unconvertible_values,
            'data_type_analysis': {
                'original_dtype': conversion_stats['original_dtype'],
                'is_categorical': pd.api.types.is_categorical_dtype(series),
                'has_mixed_types': len(set(type(x).__name__ for x in non_null_series.iloc[:10])) > 1
            }
        }
        
        # Log comprehensive information for debugging
        if error_details or warning_messages:
            context_info = {
                'total_records': total_records,
                'conversion_success_rate': f"{(conversion_stats['conversion_successes'] / max(1, series.notna().sum())):.2%}",
                'sample_values': list(series.dropna().iloc[:3].astype(str)),
                'data_type': str(series.dtype)
            }
            
            if error_details:
                self.logger.warning(f"Date validation encountered errors: {len(error_details)} errors")
                for error in error_details:
                    self.logger.debug(f"Date validation error: {error}")
            
            if warning_messages:
                self.logger.info(f"Date validation warnings: {'; '.join(warning_messages)}")
        
        self.logger.debug(f"Date validation completed: {is_valid.sum()}/{total_records} valid records")
        
        return is_valid, details
    
    def _validate_statistical_outliers(self, series: pd.Series, z_score_threshold: float = 3.0, 
                                     **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Validate using statistical outlier detection."""
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        if numeric_series.empty or numeric_series.isna().all():
            return pd.Series(True, index=series.index), {'validation_type': 'statistical_outliers'}
        
        # Calculate z-scores
        mean_val = numeric_series.mean()
        std_val = numeric_series.std()
        
        if std_val == 0:
            # No variation, all values are the same
            is_valid = pd.Series(True, index=series.index)
        else:
            z_scores = np.abs((numeric_series - mean_val) / std_val)
            is_valid = (z_scores <= z_score_threshold) | numeric_series.isna()
        
        outlier_count = (~is_valid).sum()
        
        details = {
            'z_score_threshold': z_score_threshold,
            'outlier_count': int(outlier_count),
            'mean': float(mean_val),
            'std': float(std_val),
            'validation_type': 'statistical_outliers'
        }
        
        return is_valid, details
    
    def _validate_revenue_calculation(self, series: pd.Series, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Validate revenue calculation business logic."""
        # This requires access to the full DataFrame, so we'll need to modify the approach
        # For now, just validate that revenue is positive
        numeric_series = pd.to_numeric(series, errors='coerce')
        is_valid = (numeric_series > 0) | numeric_series.isna()
        
        invalid_count = (~is_valid).sum()
        
        details = {
            'invalid_revenue_count': int(invalid_count),
            'validation_type': 'business_logic'
        }
        
        return is_valid, details
    
    def get_validation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get validation summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_results = [
            r for r in self.validation_history
            if r.timestamp >= cutoff_time
        ]
        
        if not recent_results:
            return {'message': 'No validation results in the specified time period'}
        
        # Aggregate statistics
        total_validations = len(recent_results)
        passed_validations = sum(1 for r in recent_results if r.passed)
        failed_validations = total_validations - passed_validations
        
        # Group by severity
        by_severity = {}
        for severity in ValidationSeverity:
            severity_results = [r for r in recent_results if r.severity == severity]
            if severity_results:
                by_severity[severity.value] = {
                    'total': len(severity_results),
                    'passed': sum(1 for r in severity_results if r.passed),
                    'failed': sum(1 for r in severity_results if not r.passed),
                    'avg_error_rate': sum(r.error_rate for r in severity_results) / len(severity_results)
                }
        
        # Group by field
        by_field = {}
        for result in recent_results:
            field = result.field_name
            if field not in by_field:
                by_field[field] = {'total': 0, 'passed': 0, 'failed': 0, 'avg_error_rate': 0}
            
            by_field[field]['total'] += 1
            if result.passed:
                by_field[field]['passed'] += 1
            else:
                by_field[field]['failed'] += 1
        
        # Calculate average error rates
        for field_stats in by_field.values():
            field_results = [r for r in recent_results if r.field_name == field]
            field_stats['avg_error_rate'] = sum(r.error_rate for r in field_results) / len(field_results)
        
        return {
            'time_period_hours': hours,
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'failed_validations': failed_validations,
            'success_rate': (passed_validations / total_validations) * 100,
            'by_severity': by_severity,
            'by_field': by_field,
            'recent_failures': [
                {
                    'rule_name': r.rule_name,
                    'field_name': r.field_name,
                    'severity': r.severity.value,
                    'error_rate': r.error_rate,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in sorted(recent_results, key=lambda x: x.timestamp, reverse=True)
                if not r.passed
            ][:10]  # Last 10 failures
        }
    
    def export_validation_report(self, filepath: str, hours: int = 24) -> None:
        """Export validation report to file."""
        summary = self.get_validation_summary(hours)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'validation_summary': summary,
            'registered_rules': {
                name: {
                    'field_name': rule.field_name,
                    'rule_type': rule.rule_type.value,
                    'severity': rule.severity.value,
                    'enabled': rule.enabled,
                    'threshold': rule.threshold,
                    'description': rule.description
                }
                for name, rule in self.validation_rules.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Validation report exported to {filepath}")


# Global validator instance
automated_validator = AutomatedDataValidator()


# Convenience functions
def validate_data(df: pd.DataFrame, rules: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate DataFrame using automated validator."""
    return automated_validator.validate_dataframe(df, rules)


def register_custom_validation_rule(name: str, field_name: str, rule_type: ValidationRule,
                                   severity: ValidationSeverity, validation_function: Callable,
                                   threshold: Optional[float] = None,
                                   parameters: Dict[str, Any] = None,
                                   description: str = "") -> None:
    """Register a custom validation rule."""
    automated_validator.register_validation_rule(
        name, field_name, rule_type, severity, validation_function,
        threshold, parameters, description
    )


def get_validation_summary(hours: int = 24) -> Dict[str, Any]:
    """Get validation summary."""
    return automated_validator.get_validation_summary(hours)