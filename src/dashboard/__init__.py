"""
Dashboard module for interactive data visualization.

This module provides components for creating interactive dashboards
to visualize business metrics and insights from the e-commerce data pipeline.
"""

from .dashboard_app import DashboardApp
from .data_provider import DataProvider

__all__ = ['DashboardApp', 'DataProvider']