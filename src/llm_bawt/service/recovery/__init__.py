"""Recovery for state that won't heal on its own (TASK-1232).

Each recovery action is fenced, reports what it found and what it changed, and
supports a dry run that changes nothing.
"""
