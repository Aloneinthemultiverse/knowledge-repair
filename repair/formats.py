"""Format repair: make every cell look like its column's dominant format.
Generic - learns patterns from the data itself, no per-dataset rules.

  1. placeholders (N/A, null, -) -> ''
  2. value contains a substring in the column's majority shape -> extract it
     ('12/02/2011 6:55 a.m.' in a column of '6:55 a.m.' values -> '6:55 a.m.')
  3. numbers with units: strip the unit when the column's majority has none, or
     when the unit is inconsistent ('12.0 oz' / '12 ounce' / '12.0 oz.' -> '12')
  4. redundant same-row content: 'San Francisco CA' when the state column says CA
"""
import re
from collections import Counter

PLACEHOLDERS = {"n/a", "na", "null", "none", "nan", "-", "?", "--"}
NUM_UNIT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z%][a-zA-Z%\.]*)?\s*$")


def shape(v):
    s = re.sub(r"[0-9]", "9", v)
    s = re.sub(r"[a-z]", "a", s)
    s = re.sub(r"[A-Z]", "A", s)
    s = re.sub(r"9+", "9", s)
    return re.sub(r"a+", "a", re.sub(r"A+", "A", s))


def shape_regex(sh):
    out = ""
    for ch in sh:
        out += {"9": r"\d+", "a": r"[a-z]+", "A": r"[A-Z]+"}.get(ch, re.escape(ch))
    return re.compile(out)


def _num(x):
    x = x.rstrip("0").rstrip(".") if "." in x else x
    return x or "0"


def repair_formats(df, log, min_major=0.6):
    df = df.copy()

    def put(i, col, new, rule):
        log.append({"row": i, "col": col, "before": df.at[i, col], "after": new, "rule": rule})
        df.at[i, col] = new

    for col in df.columns:
        vals = df[col]
        # 1. placeholders
        for i, v in vals.items():
            if v.strip().lower() in PLACEHOLDERS:
                put(i, col, "", "placeholder->empty")
        for i, v in df[col].items():
            if "''" in v:
                put(i, col, v.replace("''", "'"), "collapse doubled apostrophe")
            elif re.fullmatch(r"\d{1,3}(,\d{3})+", v):
                put(i, col, v.replace(",", ""), "drop thousands separator")
            elif re.fullmatch(r"-?\d+\.0", v):
                put(i, col, v[:-2], "integer written as X.0")
        nonempty = [v for v in df[col] if v]
        if not nonempty:
            continue
        # 3. numbers with units
        m = [NUM_UNIT.match(v) for v in nonempty]
        if sum(bool(x) for x in m) >= 0.9 * len(nonempty):
            units = Counter((x.group(2) or "") for x in m if x)
            top_unit, top_n = units.most_common(1)[0]
            consistent = top_n >= min_major * len(nonempty)
            if not (consistent and top_unit):          # majority bare, or units inconsistent
                bare = Counter(shape(x.group(1)) for x in m if x and not x.group(2))
                bare_shape = bare.most_common(1)[0][0] if bare and sum(bare.values()) >= min_major * len(nonempty) else None
                for i, v in df[col].items():
                    x = NUM_UNIT.match(v) if v else None
                    # a 'unit' that shrinks the number out of the column's shape is a typo ('3595x')
                    if x and x.group(2) and (bare_shape is None or shape(_num(x.group(1))) == bare_shape):
                        put(i, col, _num(x.group(1)), f"strip unit '{x.group(2)}'")
                    elif x and x.group(1).endswith(".0") and not consistent:
                        put(i, col, _num(x.group(1)), "canonical number")
                continue
        # 2. majority shape extraction
        shapes = Counter(shape(v) for v in nonempty)
        top, n = shapes.most_common(1)[0]
        if n >= min_major * len(nonempty) and len(shapes) > 1:
            rx = shape_regex(top)
            for i, v in df[col].items():
                if v and shape(v) != top:
                    hit = [h for h in rx.finditer(v)]
                    rest = (v[:hit[0].start()] + "|" + v[hit[0].end():]) if len(hit) == 1 else ""
                    left = (v[:hit[0].start()] + v[hit[0].end():]) if len(hit) == 1 else ""
                    # only strip numeric debris (date prefixes, codes); never cut words off text
                    if (len(hit) == 1 and hit[0].group(0) != v
                            and not re.search(r"[A-Za-z]", left)
                            and (not re.search(r"[A-Za-z0-9]", left)       # pure punctuation debris
                                 or ((hit[0].start() == 0 or v[hit[0].start() - 1] == " ")
                                     and (hit[0].end() == len(v) or v[hit[0].end()] == " ")))):
                        put(i, col, hit[0].group(0), f"extract majority shape {top!r}")
    # 4. redundant same-row suffix ('City ST' + state=ST)
    cols = list(df.columns)
    for a in cols:
        for b in cols:
            if a == b:
                continue
            hits = [i for i, (x, y) in enumerate(zip(df[a], df[b])) if y and x.endswith(" " + y)]
            if 0 < len(hits) < 0.5 * len(df) and len(hits) >= 5:
                known = set(df[a])
                for i in hits:
                    idx = df.index[i]
                    short = df.at[idx, a][: -(len(df.at[idx, b]) + 1)].rstrip(" ,")
                    if short in known:     # only when the shorter form is itself a real value
                        put(idx, a, short, f"drop '{b}' repeated inside '{a}'")
    return df
