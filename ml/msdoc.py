"""Pure-Python text extraction from old binary office files (no LibreOffice needed).

* OLE2 / Compound File Binary reader (the container of .doc and .xls files)
* Word 97-2003 (.doc): text is read through the piece table (Clx / PlcPcd) of the
  WordDocument stream, field codes are removed, paragraphs / table cells restored.
* Excel 97-2003 (.xls, BIFF8): cell strings are read from the shared string
  table (SST) and numbers / labels from the cell records.

Written from the public [MS-CFB], [MS-DOC] and [MS-XLS] specifications.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Dict, List, Optional

ENDOFCHAIN = 0xFFFFFFFE
FREESECT = 0xFFFFFFFF
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


class OleFile:
    def __init__(self, data: bytes):
        if data[:8] != OLE_MAGIC:
            raise ValueError("not an OLE2 compound file")
        self.data = data
        self.sec_size = 1 << struct.unpack_from("<H", data, 0x1E)[0]
        self.mini_size = 1 << struct.unpack_from("<H", data, 0x20)[0]
        n_fat = struct.unpack_from("<I", data, 0x2C)[0]
        first_dir = struct.unpack_from("<I", data, 0x30)[0]
        self.mini_cutoff = struct.unpack_from("<I", data, 0x38)[0]
        first_minifat = struct.unpack_from("<I", data, 0x3C)[0]
        first_difat, n_difat = struct.unpack_from("<II", data, 0x44)
        difat = list(struct.unpack_from("<109I", data, 0x4C))
        sect, per = first_difat, self.sec_size // 4 - 1
        for _ in range(n_difat):
            if sect in (ENDOFCHAIN, FREESECT):
                break
            vals = struct.unpack_from(f"<{per + 1}I", self._sector(sect))
            difat.extend(vals[:per]); sect = vals[per]
        fat_bytes = b"".join(self._sector(s) for s in difat[:n_fat] if s not in (FREESECT, ENDOFCHAIN))
        self.fat = list(struct.unpack(f"<{len(fat_bytes) // 4}I", fat_bytes))
        dir_bytes = self._chain(first_dir)
        self.entries: List[Dict] = []
        for off in range(0, len(dir_bytes) - 127, 128):
            e = dir_bytes[off:off + 128]
            nlen = struct.unpack_from("<H", e, 0x40)[0]
            name = e[:max(0, nlen - 2)].decode("utf-16-le", errors="ignore")
            self.entries.append({"name": name, "type": e[0x42], "start": struct.unpack_from("<I", e, 0x74)[0],
                                 "size": struct.unpack_from("<I", e, 0x78)[0]})
        root = self.entries[0]
        self.ministream = self._chain(root["start"]) if root["start"] not in (ENDOFCHAIN, FREESECT) else b""
        mf = self._chain(first_minifat) if first_minifat not in (ENDOFCHAIN, FREESECT) else b""
        self.minifat = list(struct.unpack(f"<{len(mf) // 4}I", mf))

    def _sector(self, s: int) -> bytes:
        off = (s + 1) * self.sec_size
        return self.data[off:off + self.sec_size]

    def _chain(self, start: int) -> bytes:
        out, s, seen = [], start, set()
        while s not in (ENDOFCHAIN, FREESECT) and s < len(self.fat) and s not in seen:
            seen.add(s); out.append(self._sector(s)); s = self.fat[s]
        return b"".join(out)

    def _mini_chain(self, start: int) -> bytes:
        out, s, seen = [], start, set()
        while s not in (ENDOFCHAIN, FREESECT) and s < len(self.minifat) and s not in seen:
            seen.add(s)
            out.append(self.ministream[s * self.mini_size:(s + 1) * self.mini_size]); s = self.minifat[s]
        return b"".join(out)

    def exists(self, name: str) -> bool:
        return any(e["name"].lower() == name.lower() and e["type"] == 2 for e in self.entries)

    def stream(self, name: str) -> bytes:
        for e in self.entries:
            if e["name"].lower() == name.lower() and e["type"] == 2:
                raw = self._mini_chain(e["start"]) if e["size"] < self.mini_cutoff else self._chain(e["start"])
                return raw[:e["size"]]
        raise KeyError(name)


# ------------------------------------------------------------------ Word 97-2003
def _clean_word_text(t: str) -> str:
    # fields: \x13 code \x14 result \x15  -> keep only the result
    out, stack = [], []
    for ch in t:
        if ch == "\x13":
            stack.append(True); continue            # inside field code
        if ch == "\x14":
            if stack:
                stack[-1] = False
            continue
        if ch == "\x15":
            if stack:
                stack.pop()
            continue
        if stack and stack[-1]:
            continue
        out.append(ch)
    t = "".join(out)
    t = t.replace("\x07", "\t").replace("\r", "\n").replace("\x0b", "\n").replace("\x0c", "\n\x0c")
    t = re.sub(r"[\x00-\x08\x0e-\x1f]", "", t)
    t = re.sub(r"\t{2,}", "\t", t)
    t = re.sub(r"[ \t]*\n[ \t]*", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t)


def doc_text(data: bytes) -> str:
    ole = OleFile(data)
    wd = ole.stream("WordDocument")
    ident, nfib = struct.unpack_from("<HH", wd, 0)
    if ident != 0xA5EC:
        raise ValueError("not a Word document")
    flags = struct.unpack_from("<H", wd, 0x0A)[0]
    if flags & 0x0100:
        raise ValueError("encrypted Word document")
    table = ole.stream("1Table" if flags & 0x0200 else "0Table") if nfib >= 101 else wd
    csw = struct.unpack_from("<H", wd, 32)[0]
    p = 34 + csw * 2
    cslw = struct.unpack_from("<H", wd, p)[0]
    p += 2 + cslw * 4
    blob = p + 2
    fc_clx, lcb_clx = struct.unpack_from("<II", wd, blob + 33 * 8)
    clx = table[fc_clx:fc_clx + lcb_clx]
    i = 0
    while i < len(clx) and clx[i] == 0x01:                   # Prc (formatting) entries
        cb = struct.unpack_from("<h", clx, i + 1)[0]
        i += 3 + cb
    if i >= len(clx) or clx[i] != 0x02:
        raise ValueError("piece table not found")
    lcb = struct.unpack_from("<I", clx, i + 1)[0]
    plc = clx[i + 5:i + 5 + lcb]
    n = (lcb - 4) // 12
    cps = struct.unpack_from(f"<{n + 1}I", plc, 0)
    parts = []
    for k in range(n):
        fc = struct.unpack_from("<I", plc, (n + 1) * 4 + k * 8 + 2)[0]
        length = cps[k + 1] - cps[k]
        if fc & 0x40000000:
            off = (fc & ~0x40000000) // 2
            parts.append(wd[off:off + length].decode("cp1252", errors="ignore"))
        else:
            parts.append(wd[fc:fc + 2 * length].decode("utf-16-le", errors="ignore"))
    return _clean_word_text("".join(parts))


# ------------------------------------------------------------------ Excel 97-2003
def xls_text(data: bytes) -> List[str]:
    """One text block per worksheet (rows joined by newlines, cells by ' | ')."""
    ole = OleFile(data)
    wb = ole.stream("Workbook") if ole.exists("Workbook") else ole.stream("Book")
    recs = []
    pos = 0
    while pos + 4 <= len(wb):
        rid, ln = struct.unpack_from("<HH", wb, pos)
        recs.append((rid, wb[pos + 4:pos + 4 + ln]))
        pos += 4 + ln
    # --- shared string table (SST + CONTINUE records)
    sst: List[str] = []
    for idx, (rid, body) in enumerate(recs):
        if rid != 0x00FC:
            continue
        chunks = [body]
        j = idx + 1
        while j < len(recs) and recs[j][0] == 0x003C:
            chunks.append(recs[j][1]); j += 1
        total = struct.unpack_from("<I", body, 4)[0]
        ci, p = 0, 8
        cur = chunks[0]
        for _ in range(total):
            if p >= len(cur):
                ci += 1
                if ci >= len(chunks):
                    break
                cur, p = chunks[ci], 0
            nch = struct.unpack_from("<H", cur, p)[0]; flags = cur[p + 2]; p += 3
            rt = ext = 0
            if flags & 0x08:
                rt = struct.unpack_from("<H", cur, p)[0]; p += 2
            if flags & 0x04:
                ext = struct.unpack_from("<I", cur, p)[0]; p += 4
            wide = flags & 0x01
            s, left = [], nch
            while left > 0:
                if p >= len(cur):
                    ci += 1
                    if ci >= len(chunks):
                        break
                    cur = chunks[ci]; wide = cur[0] & 0x01; p = 1
                avail = (len(cur) - p) // (2 if wide else 1)
                take = min(left, avail)
                if wide:
                    s.append(cur[p:p + 2 * take].decode("utf-16-le", errors="ignore")); p += 2 * take
                else:
                    s.append(cur[p:p + take].decode("latin-1", errors="ignore")); p += take
                left -= take
            skip = rt * 4 + ext
            while skip > 0:
                if p >= len(cur):
                    ci += 1
                    if ci >= len(chunks):
                        break
                    cur, p = chunks[ci], 0
                step = min(skip, len(cur) - p); p += step; skip -= step
            sst.append("".join(s))
        break
    # --- cells per sheet (a new sheet starts with a BOF record of type worksheet)
    sheets: List[Dict] = []
    cells = None
    for rid, body in recs:
        if rid == 0x0809 and len(body) >= 4 and struct.unpack_from("<H", body, 2)[0] == 0x0010:
            cells = {}; sheets.append(cells)
        elif cells is None:
            continue
        elif rid == 0x00FD and len(body) >= 10:                   # LABELSST
            r, c, _, k = struct.unpack_from("<HHHI", body, 0)
            if k < len(sst):
                cells[(r, c)] = sst[k]
        elif rid == 0x0203 and len(body) >= 14:                   # NUMBER
            r, c, _, v = struct.unpack_from("<HHHd", body, 0)
            cells[(r, c)] = f"{v:g}"
        elif rid == 0x027E and len(body) >= 10:                   # RK
            r, c, _, rk = struct.unpack_from("<HHHI", body, 0)
            v = struct.unpack("<d", struct.pack("<Q", (rk & 0xFFFFFFFC) << 32))[0] if not rk & 2 else float(rk >> 2)
            cells[(r, c)] = f"{v / 100 if rk & 1 else v:g}"
    out = []
    for cells in sheets:
        rows: Dict[int, Dict[int, str]] = {}
        for (r, c), v in cells.items():
            if str(v).strip():
                rows.setdefault(r, {})[c] = str(v).strip()
        lines = [" | ".join(rows[r][c] for c in sorted(rows[r])) for r in sorted(rows)]
        if lines:
            out.append("\n".join(lines))
    return out


def extract_binary_office(path: Path) -> Optional[List[str]]:
    """Pages (Word: split on page breaks / ~3000 chars, Excel: one per sheet) or None if not parsable."""
    data = Path(path).read_bytes()
    if data[:8] != OLE_MAGIC:
        return None
    try:
        ole_names = {e["name"].lower() for e in OleFile(data).entries}
        if "worddocument" in ole_names:
            text = doc_text(data)
            pages, cur = [], ""
            for part in text.split("\x0c"):
                for para in part.split("\n"):
                    if len(cur) + len(para) > 3000 and cur:
                        pages.append(cur); cur = ""
                    cur += para + "\n"
                if cur.strip():
                    pages.append(cur); cur = ""
            if cur.strip():
                pages.append(cur)
            return [p.strip() for p in pages if p.strip()]
        if "workbook" in ole_names or "book" in ole_names:
            return xls_text(data)
    except Exception:
        return None
    return None
