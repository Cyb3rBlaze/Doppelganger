"""Shared helper objects for pytest-based tests."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any


class AsyncSpy:
    """Minimal awaitable test double for async tests."""

    def __init__(self, result: Any = None, side_effect: Any = None) -> None:
        self._result = result
        self._side_effect = side_effect
        self.await_count = 0
        self.await_args = SimpleNamespace(args=(), kwargs={})

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.await_count += 1
        self.await_args = SimpleNamespace(args=args, kwargs=kwargs)

        effect = self._consume_side_effect()
        if effect is not None:
            if isinstance(effect, BaseException):
                raise effect
            if callable(effect):
                return effect(*args, **kwargs)
            return effect

        return self._result

    def _consume_side_effect(self) -> Any:
        if self._side_effect is None:
            return None
        if isinstance(self._side_effect, Iterator):
            return next(self._side_effect)
        if isinstance(self._side_effect, list):
            self._side_effect = iter(self._side_effect)
            return next(self._side_effect)
        effect = self._side_effect
        self._side_effect = None
        return effect

    def assert_awaited_once(self) -> None:
        assert self.await_count == 1

    def assert_awaited_once_with(self, *args: Any, **kwargs: Any) -> None:
        self.assert_awaited_once()
        assert self.await_args.args == args
        assert self.await_args.kwargs == kwargs

    def assert_not_called(self) -> None:
        assert self.await_count == 0
