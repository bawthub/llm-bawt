"""Assemble a contiguous persisted partial from potentially reordered text events.

The wire protocol's text_offset is measured in UTF-16 code units (the browser's
string indexing unit), while Python string indices count Unicode code points.
Never append across an unknown gap: doing so invents positions for missing text.
"""


class PartialText:
    def __init__(self) -> None:
        self._prefix = b""
        self._waiting: dict[int, bytes] = {}

    def add(self, text: str, offset: int | None) -> str:
        data = text.encode("utf-16-le", errors="surrogatepass")
        if not isinstance(offset, int) or offset < 0:
            offset = len(self._prefix) // 2
        if offset > len(self._prefix) // 2:
            self._waiting[offset] = data
        else:
            self._splice(offset, data)
            # A newly filled gap may unlock several already-seen chunks.
            while True:
                eligible = [n for n in self._waiting if n <= len(self._prefix) // 2]
                if not eligible:
                    break
                for n in sorted(eligible):
                    self._splice(n, self._waiting.pop(n))
        return self._prefix.decode("utf-16-le", errors="surrogatepass")

    def _splice(self, offset: int, data: bytes) -> None:
        start = offset * 2
        end = start + len(data)
        # Replays/overlaps may replace known bytes but cannot erase the tail.
        self._prefix = self._prefix[:start] + data + self._prefix[end:]
