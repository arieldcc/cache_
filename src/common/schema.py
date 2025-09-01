# Unified schema for traces
from dataclasses import dataclass
@dataclass(frozen=True)
class TraceCols:
    ts: str = "ts"
    key: str = "key"
    size: str = "size_bytes"
    op: str = "op"
    ttl: str = "ttl"
