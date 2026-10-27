"""Recurring context-waste report (TASK-1585)."""

from .queries import Scope
from .report import SETTING_KEY, build_report, resolve_thresholds
from .scope import caller_scope, operator_scope

__all__ = ["SETTING_KEY", "Scope", "build_report", "caller_scope", "operator_scope",
           "resolve_thresholds"]
