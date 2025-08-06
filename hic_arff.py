import numpy as np
import pandas as pd
from pathlib import Path
import re, io
from collections import Counter, defaultdict


def _make_unique(names):
    seen = {}
    out  = []
    for n in names:
        if n in seen:
            seen[n] += 1
            n = f"{n}.{seen[n]}"
        else:
            seen[n] = 0
        out.append(n)
    return out

def read_arff(path: Path, na='?') -> pd.DataFrame:
    head, raw = [], ""
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip() == '' or ln.strip().startswith('%'):
                continue
            if ln.lower().startswith('@attribute'):
                head.append(re.split(r'\s+', ln, 2)[1])
            elif ln.lower().startswith('@data'):
                break
        raw = f.read()

    head = _make_unique(head)        # ← ensure uniqueness
    return pd.read_csv(
        io.StringIO(raw),
        names=head,
        na_values=na,
        keep_default_na=True,
        comment='%'
    )

def child_parent_map(arff_path: Path, Y_all=None) -> dict[str, str]:
    def _read_hier_block(p: Path) -> str:
        lines = []
        with open(p, encoding="utf-8") as f:
            in_attr = False
            brace = 0
            for raw in f:
                ln = raw.strip()
                if not in_attr:
                    if ln.lower().startswith("@attribute") and "hierarchical" in ln.lower():
                        in_attr = True
                        part = ln.split("hierarchical", 1)[1].strip()
                        lines.append(part)
                        brace += part.count("{") - part.count("}")
                        if brace == 0 and "{" not in part:
                            break
                        if brace == 0 and "}" in part:
                            break
                else:
                    lines.append(ln)
                    brace += ln.count("{") - ln.count("}")
                    if brace <= 0:
                        break
        raw_block = " ".join(lines)
        raw_block = re.sub(r"^\s*\{(.*)\}\s*$", r"\1", raw_block)  # drop outer braces if any
        return raw_block

    def _parse_paths(raw_block: str):
        if not raw_block:
            return []
        parts = [p.strip().strip("'\"") for p in raw_block.split(",") if p.strip()]
        paths = []
        for p in parts:
            nodes = [n.strip().strip("'\"") for n in p.split("/") if n.strip()]
            if nodes:
                paths.append(nodes)
        return paths

    # 1) Parse hierarchical attribute (DAG edges with stable header order)
    raw_block = _read_hier_block(arff_path)
    arff_paths = _parse_paths(raw_block)

    c2p_multi: dict[str, set[str]] = defaultdict(set)
    first_order: dict[tuple[str, str], int] = {}
    ord_idx = 0
    for path in arff_paths:
        for pa, ch in zip(path, path[1:]):
            c2p_multi[ch].add(pa)
            if (pa, ch) not in first_order:
                first_order[(pa, ch)] = ord_idx
                ord_idx += 1

    # 2) Optional: count observed (parent->child) frequencies from Y_all
    pair_freq = Counter()
    if Y_all is not None:
        for s in Y_all:
            for path in s:
                for pa, ch in zip(path, path[1:]):
                    pair_freq[(pa, ch)] += 1

    # 3) Deterministic choice: highest freq -> earliest header order -> lexicographic
    single_parent: dict[str, str] = {}
    for ch, parents in c2p_multi.items():
        if not parents:
            continue
        if len(parents) == 1:
            single_parent[ch] = next(iter(parents))
            continue
        candidates = []
        for pa in parents:
            freq = pair_freq[(pa, ch)]
            order = first_order.get((pa, ch), 1_000_000_000)
            candidates.append((-freq, order, pa))  # sort ascending: more freq, earlier order, then lexicographic pa
        chosen = sorted(candidates)[0][2]
        single_parent[ch] = chosen

    return single_parent


def parent_child_map(c2p: dict[str, str]) -> dict[str, list[str]]:
    p2c = defaultdict(list)
    for ch, pa in c2p.items():
        p2c[pa].append(ch)
    return dict(p2c)


def parse_cell(cell: str | float, c2p: dict[str, str], root="root") -> list[list[str]]:
    if pd.isna(cell) or str(cell).strip() in {"", "?"}:
        return []
    paths = []
    for term in str(cell).split('@'):
        path = [term]
        while path[0] in c2p:
            path.insert(0, c2p[path[0]])
        if path[0] != root:
            path.insert(0, root)
        paths.append(path)
    return paths

def load_split(base: str, split: str, c2p: dict[str, str], data_dir: Path):
    df = read_arff(data_dir / f"{base}.{split}.arff")
    y = np.array(df['class'].apply(lambda c: parse_cell(c, c2p)).tolist(),
                 dtype=object)
    X = (df.drop(columns="class")
           .apply(pd.to_numeric, errors="coerce")
           .astype(np.float32))

    X = X.dropna(axis=1, how="all")
    
    if X.shape[1] == 0: X["const0"] = 0.0

    return X, y