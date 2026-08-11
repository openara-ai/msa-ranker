"""ULID generation — the per-event idempotency anchor (spec 01)."""

from __future__ import annotations

import os
import time

# Crockford base32 (no I, L, O, U).
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid(ms: int | None = None) -> str:
    """Return a 26-char ULID: 48-bit millisecond timestamp + 80 random bits.

    Lexicographically sortable by time. Uniqueness comes from 80 bits of
    `os.urandom`; this is the SoR idempotency anchor, not a security token.
    """
    if ms is None:
        ms = int(time.time() * 1000)
    value = (ms << 80) | int.from_bytes(os.urandom(10), "big")
    chars = []
    for _ in range(26):
        chars.append(_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))
