from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import pickle
from typing import Any, Iterator, Mapping


class MulticonductorStudyError(RuntimeError):
    """Base class for canonical multiconductor study failures."""


class InvalidStudyInput(MulticonductorStudyError, ValueError):
    """Raised when required network or study inputs are missing or invalid."""


class StudyNotConverged(MulticonductorStudyError):
    """Raised when a study cannot converge to a valid solution."""


class StudyStatus(str, Enum):
    CONVERGED = "converged"
    NONCONVERGED = "nonconverged"
    ISLANDED = "islanded"
    INVALID = "invalid"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TerminalRef:
    """Reference to one electrical bus terminal."""

    bus: int
    phase: int
    side: str | None = None

    def as_tuple(self) -> tuple[int, int]:
        return (self.bus, self.phase)


@dataclass(frozen=True, slots=True)
class ElementRef:
    """Reference to an element row or one conductor row of an element."""

    table: str
    index: Any
    circuit: int | None = None
    bus: int | None = None
    side: str | None = None

    @property
    def element_type(self) -> str:
        return self.table

    @property
    def element_index(self) -> Any:
        return self.index


@dataclass(frozen=True, slots=True)
class StudySnapshot:
    payload: bytes
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(cls, net: Any, **metadata: Any) -> "StudySnapshot":
        return cls(payload=pickle.dumps(net, protocol=pickle.HIGHEST_PROTOCOL), metadata=dict(metadata))

    def restore_into(self, net: Any) -> Any:
        restored = pickle.loads(self.payload)
        current_keys = list(net.keys())
        for key in current_keys:
            if key not in restored:
                del net[key]
        for key, value in restored.items():
            net[key] = value
        return net


@contextmanager
def preserved_network(net: Any, **metadata: Any) -> Iterator[StudySnapshot]:
    snapshot = StudySnapshot.capture(net, **metadata)
    try:
        yield snapshot
    finally:
        snapshot.restore_into(net)
