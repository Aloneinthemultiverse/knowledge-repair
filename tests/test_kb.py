"""Tests for the knowledge-base repair tool (python -m pytest tests/test_kb.py)."""
import os
import subprocess
import sys

import pandas as pd
import pytest

from kb.corrupt import corrupt
from kb.dq import dq
from kb.load import load
from kb.repair import KBRepairer
from kb.score import score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def run():
    clean = load()
    dirty, truth, log = corrupt(clean, seed=7)
    snapshot = {t: df.copy() for t, df in dirty.items()}
    rep = KBRepairer(dirty).run()
    return clean, dirty, snapshot, truth, log, rep


def test_corruption_is_reproducible():
    clean = load()
    a = corrupt(clean, seed=3)[2]
    b = corrupt(clean, seed=3)[2]
    assert a.equals(b)


def test_input_is_never_modified(run):
    _, dirty, snapshot, *_ = run
    for t in dirty:
        pd.testing.assert_frame_equal(dirty[t], snapshot[t])


def test_every_record_maps_to_an_entity(run):
    _, dirty, _, _, _, rep = run
    for t, idc in (("people", "person_id"), ("places", "place_id"), ("events", "event_id")):
        assert set(dirty[t][idc]) == set(rep.canon[t])
        out = rep.output()[t]
        sources = {s for v in out.source_records for s in v.split(",")}
        assert sources == set(dirty[t][idc])          # lineage: no record lost, none invented


def test_lineage_covers_every_repaired_value(run):
    *_, rep = run
    lin = pd.DataFrame(rep.lineage)
    for t, n_cols in (("people", 6), ("places", 2), ("events", 4)):
        assert len(lin[lin.table == t]) == len(rep.output()[t]) * n_cols


def test_every_action_is_explained(run):
    *_, rep = run
    for a in rep.actions:
        assert a["rule"] and a["explanation"] and a["status"] in ("applied", "needs_review", "flagged_only")


def test_quality_against_ground_truth(run):
    clean, dirty, _, truth, log, rep = run
    m = score(rep, clean, truth, log)
    harm, cells = map(int, m["harm_total"].split("/"))
    assert harm / cells < 0.002                         # repair breaks almost nothing
    assert float(m["relationships_after_P/R/F1"].split("/")[2]) >= 0.98
    assert float(m["people_dup_P/R/F1"].split("/")[0]) >= 0.99   # no wrong person merges


def test_quality_score_improves_without_ground_truth(run):
    _, dirty, *_, rep = run
    assert dq(rep.output())["DQ"] > dq(dirty)["DQ"]


def test_repairing_repaired_output_changes_little(run):
    *_, rep = run
    again = KBRepairer({t: df.drop(columns=[c for c in ("source_records", "from_row") if c in df], errors="ignore")
                        for t, df in rep.output().items()}).run()
    merges = [a for a in again.actions if a["kind"] == "MERGE"]
    assert len(merges) <= 3                            # output is already deduplicated


def test_cli_run_writes_all_outputs(run, tmp_path):
    _, dirty, *_ = run
    src = tmp_path / "in"
    src.mkdir()
    for t, df in dirty.items():
        df.to_csv(src / f"{t}.csv", index=False)
    out = tmp_path / "out"
    subprocess.run([sys.executable, "-m", "kb", "run", str(src), "--out", str(out)], cwd=ROOT, check=True,
                   capture_output=True)
    for f in ("repaired_people.csv", "repaired_relationships.csv", "lineage.csv", "issues.csv",
              "actions.csv", "quality.json", "report.html"):
        assert (out / f).exists(), f
