"""Execution backends. The graph only ever talks to the :class:`Executor` protocol."""

from amt.executor.base import Executor, ExecutorError
from amt.executor.local_mock import LocalMockExecutor

__all__ = ["Executor", "ExecutorError", "LocalMockExecutor"]
