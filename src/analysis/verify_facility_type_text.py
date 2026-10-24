#!/usr/bin/env python3
"""Check every number in the proposed facility-type text against the generated outputs.

Reads ``table_numbers.json`` and ``join_report.json`` written by
``facility_type_electricity_services.py`` and the proposed text
(``proposed_manuscript_text.md``).  Each number in the Results and Implications
blocks must match a value recomputed here from the outputs, and every number in
those blocks must be covered by a check.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}"


def blocks(text: str) -> dict[str, str]:
    out = {}
    for name in ("Results addition", "Implications for risk mitigation"):
        section = text.split(f"## {name}", 1)[1].split("\n## ", 1)[0]
        quoted = [line[1:].strip() for line in section.splitlines() if line.startswith(">")]
        out[name] = " ".join(quoted).replace("$^{-1}$", "")  # unit exponent, not a reported number
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    numbers = json.loads((args.output_dir / "table_numbers.json").read_text())
    text = (args.output_dir / "proposed_manuscript_text.md").read_text()
    t, e = numbers["facility_type"], numbers["equipment"]
    h, c, p = t["Hospitals"], t["Health centers"], t["Health posts"]
    fridge, oxygen, incubator = e["Vaccine refrigerator"], e["Oxygen concentrator"], e["Neonatal incubator"]
    mean = lambda s: f"{s['mean']:.4f}"
    median = lambda s: f"{s['median']:.4f}"

    expected = {
        "Results addition": [
            str(h["main_source"]["Public grid"]), str(h["facilities"]),
            pct(c["main_source"]["Public grid"], c["facilities"]), pct(p["main_source"]["Public grid"], p["facilities"]),
            pct(c["main_source"]["Solar"], c["facilities"]), pct(p["main_source"]["Solar"], p["facilities"]),
            pct(p["main_source"]["No electricity reported"], p["facilities"]),
            pct(h["generator_reported"], h["facilities"]), pct(c["generator_reported"], c["facilities"]),
            pct(p["generator_reported"], p["facilities"]),
            mean(h["index_grid"]), median(h["index_grid"]), mean(p["index_grid"]),
            mean(h["index_all"]), mean(c["index_all"]), mean(p["index_all"]),
            str(oxygen["facilities"]), str(incubator["facilities"]),
            pct(oxygen["main_source"]["Public grid"], oxygen["facilities"]),
            pct(incubator["main_source"]["Public grid"], incubator["facilities"]),
            f"{fridge['facilities']:,}", pct(fridge["main_source"]["Public grid"], fridge["facilities"]),
            mean(oxygen["index_grid"]), mean(incubator["index_grid"]), mean(fridge["index_grid"]),
            pct(oxygen["generator_reported"], oxygen["facilities"]),
            pct(fridge["generator_reported"], fridge["facilities"]),
        ],
        "Implications for risk mitigation": [pct(fridge["generator_reported"], fridge["facilities"])],
    }
    # Statements checked as relations rather than single numbers.
    relations = {
        "hospital and health-center grid means round to the same value": mean(h["index_grid"]) == mean(c["index_grid"]),
        "hospital and health-center grid medians round to the same value": median(h["index_grid"]) == median(c["index_grid"]),
        "health-post grid mean is lower": p["index_grid"]["mean"] < min(h["index_grid"]["mean"], c["index_grid"]["mean"]),
        "oxygen and incubator generator shares both round to 49%":
            pct(oxygen["generator_reported"], oxygen["facilities"]) == pct(incubator["generator_reported"], incubator["facilities"]) == "49",
        "no health posts record oxygen concentrators or incubators":
            oxygen["by_type"]["Health posts"] == incubator["by_type"]["Health posts"] == 0,
        "grid-supplied mean index exceeds solar-supplied (all facilities)":
            t["All"]["index_grid"]["mean"] > t["All"]["index_solar"]["mean"],
        "incubator and refrigerator grid means round to the same value": mean(incubator["index_grid"]) == mean(fridge["index_grid"]),
    }
    ok = True
    for name, block in blocks(text).items():
        found = Counter(re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", block))
        want = Counter(expected[name])
        if found != want:
            ok = False
            print(f"{name}: MISMATCH\n  in text only: {dict(found - want)}\n  expected only: {dict(want - found)}")
        else:
            print(f"{name}: all {sum(found.values())} numbers match the outputs")
    for label, result in relations.items():
        ok &= bool(result)
        print(f"{'OK ' if result else 'FAIL'} {label}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
