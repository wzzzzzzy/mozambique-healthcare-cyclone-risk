#!/usr/bin/env python3
"""Electricity disruption by facility type and by recorded electricity-dependent equipment.

Descriptive table for hospitals, health centers and health posts, joining the
SARA 2018 facility survey to the existing cyclone simulation outputs.  Nothing
is re-simulated.

Facility records
----------------
* ``--sara-processed``: the facility layer used by the model (``SARA_processed``);
  its row order defines the model node IDs (``node/46774825{row:04d}``) and
  ``facility_i`` is the SARA facility number (Q001).
* ``--sara-raw``: the SARA 2018 raw export (header on the third row).  Records
  are joined on Q001 = ``facility_i``; facility type, electricity and equipment
  items are read from the raw export.

Facility type (Q007): hospitals 11-17 (rural, district, general, provincial,
specialized, central, military), health centers 3-8 (rural I/II, urban A/B/C,
private), health posts 1.

Electricity (Q408 any source; Q410 main source: 1 public grid, 2 generator,
3 solar, 4 gas, 5 private operator grid, 96 other; Q411 alternative source;
Q413 generator functioning, asked when a generator is the main or alternative
source).  "Generator reported" means Q410 or Q411 = 2; Q413 = 1 is the
respondent's report that it works, not evidence of operational back-up.

Equipment indicators (only where recorded; never inferred from facility type):
* vaccine refrigerator available and functional (Q1108 = 1) and powered by
  grid or generator electricity (Q1109 = 1); asked where immunisation services
  are offered (Q1100 = 1).  Solar, gas and kerosene refrigerators are counted in
  the join report but not tabulated: a solar refrigerator usually has its own
  panel, which the facility-level index does not describe;
* oxygen concentrator observed and functional (Q500_10A = 1, Q500_10B = 1);
  general outpatient equipment, asked of all facilities;
* neonatal incubator in the delivery area observed and functional
  (Q1005_10A = 1, Q1005_10B = 1); asked where delivery services are offered
  (Q1000 = 1).

Annualized electricity disruption index, s_h = (1/T) sum_z s_h(z), T = 10,000 yr
(mean over the 2,703 synthetic events x 2,703/10,000), where s_h(z) is the
expected fractional supply shortfall at facility h in event z:
* grid-supplied (Q410 = 1): hybrid power model, ``--public-summary``; -1 marks
  a facility outside the modelled grid (unavailable, excluded);
* solar-supplied: independent solar model, ``solar_only`` rows of
  ``--solar-per-hospital`` (the Figure 2 / Figure 4 solar baseline).
Facilities with other main sources, no electricity, or no coordinates have no
estimate and are excluded from index statistics (not set to zero).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ANNUAL = 2703 / 10_000
TYPES = {"Hospitals": set(range(11, 18)), "Health centers": set(range(3, 9)), "Health posts": {1}}
PROCESSED_TYPE = {"Hospitals": "hospital", "Health centers": "health center", "Health posts": "health post"}
INDICATORS = {
    "Vaccine refrigerator": "fridge_grid",
    "Oxygen concentrator": "oxygen",
    "Neonatal incubator": "incubator",
}
MIN_N = 10  # supply-specific statistics are shown only for at least this many facilities


def code(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def load(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    report: dict = {}
    processed = gpd.read_file(args.sara_processed)
    processed["node_id"] = [f"node/46774825{i:04d}" for i in processed.index]
    raw = pd.read_excel(args.sara_raw, header=2)
    report["records"] = {"processed": len(processed), "raw": len(raw)}
    if not processed["facility_i"].is_unique or not raw["Q001"].is_unique:
        raise ValueError("facility numbers are not unique")

    merged = processed.drop(columns="geometry").merge(raw, left_on="facility_i", right_on="Q001",
                                                      how="outer", indicator=True)
    report["join_sara_processed_to_raw"] = merged["_merge"].value_counts().to_dict()
    for left, right in (("ele_main_s", "Q410"), ("ele_sec_so", "Q411"), ("ele_any_so", "Q408")):
        a, b = code(merged, left), code(merged, right)
        report.setdefault("processed_vs_raw_disagreements", {})[f"{left}/{right}"] = int(
            (~((a == b) | (a.isna() & b.isna()))).sum())
    f = merged[merged["_merge"] == "both"].drop(columns="_merge").copy()
    f["has_coordinates"] = processed.set_index("node_id").geometry.notna().reindex(f["node_id"]).to_numpy()

    q007 = code(f, "Q007")
    f["facility_type"] = None
    for label, codes in TYPES.items():
        f.loc[q007.isin(codes), "facility_type"] = label
    report["facility_type_unclassified"] = int(f["facility_type"].isna().sum())
    report["facility_type_vs_processed_type_disagreements"] = int(
        (f["facility_type"].map(PROCESSED_TYPE) != f["type"]).sum())

    main, backup = code(f, "Q410"), code(f, "Q411")
    any_source = code(f, "Q408")
    f["main_source"] = np.select(
        [main.eq(1), main.eq(3), main.isin([2, 4, 5, 96]), any_source.eq(2)],
        ["Public grid", "Solar", "Generator, gas or other", "No electricity reported"], default="Not recorded")
    report["electricity_not_recorded"] = f.loc[f["main_source"] == "Not recorded",
                                               ["facility_i", "name", "facility_type"]].astype(str).to_dict("records")
    f["generator_reported"] = main.eq(2) | backup.eq(2)
    f["generator_functioning"] = code(f, "Q413").eq(1)
    f["generator_fuel_today"] = code(f, "Q414").eq(1)
    report["generator_functioning_without_generator_reported"] = int(
        (f["generator_functioning"] & ~f["generator_reported"]).sum())

    fridge_ok, fridge_power = code(f, "Q1108").eq(1), code(f, "Q1109")
    f["fridge_grid"] = fridge_ok & fridge_power.eq(1)
    f["fridge_solar"] = fridge_ok & fridge_power.eq(2)
    f["oxygen"] = code(f, "Q500_10A").eq(1) & code(f, "Q500_10B").eq(1)
    f["incubator"] = code(f, "Q1005_10A").eq(1) & code(f, "Q1005_10B").eq(1)
    report["equipment_items"] = {
        "immunisation_offered_Q1100": int(code(f, "Q1100").eq(1).sum()),
        "vaccine_fridge_item_answered_Q1108": int(code(f, "Q1108").notna().sum()),
        "vaccine_fridge_functional_Q1108_1": int(fridge_ok.sum()),
        "vaccine_fridge_functional_by_energy_Q1109": {
            str(k): int(v) for k, v in fridge_power[fridge_ok].value_counts(dropna=False).sort_index().items()},
        "vaccine_fridge_power_not_24_7_Q1110_2": {
            "grid_or_generator": int((f["fridge_grid"] & code(f, "Q1110").eq(2)).sum()),
            "solar": int((f["fridge_solar"] & code(f, "Q1110").eq(2)).sum())},
        "oxygen_item_answered_Q500_10A": int(code(f, "Q500_10A").notna().sum()),
        "oxygen_concentrator_observed_Q500_10A_1": int(code(f, "Q500_10A").eq(1).sum()),
        "oxygen_concentrator_reported_not_seen_Q500_10A_2": int(code(f, "Q500_10A").eq(2).sum()),
        "delivery_offered_Q1000": int(code(f, "Q1000").eq(1).sum()),
        "incubator_item_answered_Q1005_10A": int(code(f, "Q1005_10A").notna().sum()),
        "incubator_observed_Q1005_10A_1": int(code(f, "Q1005_10A").eq(1).sum()),
        "incubator_reported_not_seen_Q1005_10A_2": int(code(f, "Q1005_10A").eq(2).sum()),
    }

    public = pd.read_csv(args.public_summary)
    tracks = [c for c in public.columns if c not in ("node_id", "facility_i")]
    values = public[tracks].to_numpy(float)
    unavailable = (values < 0).all(axis=1)
    if ((values < 0).any(axis=1) & ~unavailable).any():
        raise ValueError("partially unavailable hybrid results")
    public["grid_index"] = np.where(unavailable, np.nan, values.clip(min=0).mean(axis=1) * ANNUAL)
    solar = pd.read_csv(args.solar_per_hospital)
    solar = solar[solar["baseline_source"] == "solar_only"].copy()
    if (solar["baseline_mean"] < 0).any():
        raise ValueError("unavailable solar estimates")
    solar["solar_index"] = solar["baseline_mean"] * ANNUAL
    for name, frame in (("public_summary", public), ("solar_per_hospital", solar)):
        check = frame[["node_id", "facility_i"]].merge(f[["node_id", "facility_i"]], on="node_id",
                                                        how="left", suffixes=("", "_sara"))
        report.setdefault("output_id_check", {})[name] = {
            "rows": len(frame), "node_not_in_sara": int(check["facility_i_sara"].isna().sum()),
            "facility_number_mismatch": int((check["facility_i"] != check["facility_i_sara"]).sum())}
    f = f.merge(public[["facility_i", "grid_index"]], on="facility_i", how="left")
    f = f.merge(solar[["facility_i", "solar_index"]], on="facility_i", how="left")
    f["in_hybrid_model"] = f["facility_i"].isin(public["facility_i"])
    f["in_solar_model"] = f["facility_i"].isin(solar["facility_i"])
    if (f["grid_index"].notna() & f["solar_index"].notna()).any():
        raise ValueError("facility has both a grid and a solar estimate")
    f["model"] = np.select([f["in_hybrid_model"], f["in_solar_model"]], ["grid", "solar"], default="none")
    f["index"] = f["grid_index"].fillna(f["solar_index"])

    accounting = {}
    for label, d in f.groupby("main_source"):
        accounting[label] = {
            "facilities": len(d), "no_coordinates": int((~d["has_coordinates"]).sum()),
            "in_hybrid_model": int(d["in_hybrid_model"].sum()),
            "hybrid_unavailable": int((d["in_hybrid_model"] & d["grid_index"].isna()).sum()),
            "in_solar_model": int(d["in_solar_model"].sum()), "evaluated": int(d["index"].notna().sum())}
    report["cohort_accounting_by_main_source"] = accounting
    report["solar_model_with_non_solar_main_source"] = f.loc[
        f["in_solar_model"] & (f["main_source"] != "Solar"),
        ["facility_i", "name", "main_source", "Q411"]].astype(str).to_dict("records")
    report["hybrid_unavailable_facilities"] = f.loc[
        f["in_hybrid_model"] & f["grid_index"].isna(), ["facility_i", "name", "facility_type", "region"]
    ].astype(str).to_dict("records")
    return f, report


def stats(values: pd.Series) -> dict:
    v = values.dropna().to_numpy(float)
    if len(v) == 0:
        return {"n": 0}
    q1, median, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(median), "q1": float(q1), "q3": float(q3)}


def summarise(d: pd.DataFrame) -> dict:
    n = len(d)
    out = {
        "facilities": n,
        "by_type": {t: int((d["facility_type"] == t).sum()) for t in TYPES},
        "main_source": {s: int((d["main_source"] == s).sum()) for s in
                        ("Public grid", "Solar", "Generator, gas or other", "No electricity reported",
                         "Not recorded")},
        "generator_reported": int(d["generator_reported"].sum()),
        "generator_functioning": int(d["generator_functioning"].sum()),
        "index_all": stats(d["index"]),
        "index_grid": stats(d.loc[d["model"] == "grid", "index"]),
        "index_solar": stats(d.loc[d["model"] == "solar", "index"]),
    }
    return out


def pct(count: int, total: int) -> str:
    return f"{count:,} ({100 * count / total:.0f}%)" if total else "0"


def fmt_stats(s: dict, show: bool = True) -> tuple[str, str]:
    if s["n"] == 0:
        return "--", "--"
    if not show:
        return "n < 10", "n < 10"
    return f"{s['mean']:.4f}", f"{s['median']:.4f} ({s['q1']:.4f}–{s['q3']:.4f})"


def rows_for(summary: dict, with_types: bool, supplies=("index_grid", "index_solar")) -> list[tuple[str, str]]:
    n = summary["facilities"]
    rows = [("Facilities, n", f"{n:,}")]
    if with_types:
        rows.append(("  hospitals / health centers / health posts",
                     " / ".join(f"{summary['by_type'][t]:,}" for t in TYPES)))
    for source, label in (("Public grid", "Main source: public grid"), ("Solar", "Main source: solar"),
                          ("Generator, gas or other", "Main source: generator, gas or other"),
                          ("No electricity reported", "No electricity reported")):
        rows.append((label, pct(summary["main_source"][source], n)))
    rows.append(("Generator reported (main or alternative)", pct(summary["generator_reported"], n)))
    rows.append(("  reported functioning", f"{summary['generator_functioning']:,}"))
    rows.append(("Evaluated, n", f"{summary['index_all']['n']:,}"))
    mean, med = fmt_stats(summary["index_all"])
    rows += [("Index, mean (yr⁻¹)", mean), ("  median (IQR)", med)]
    for key, label in (("index_grid", "Grid-supplied"), ("index_solar", "Solar-supplied")):
        if key not in supplies:
            continue
        s = summary[key]
        rows.append((f"{label}, evaluated n", f"{s['n']:,}"))
        mean, med = fmt_stats(s, s["n"] >= MIN_N)
        rows += [("  index, mean", mean), ("  median (IQR)", med)]
    return rows


def to_latex(panels: list[tuple[str, list[str], list[list[tuple[str, str]]]]]) -> str:
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule"]
    for p, (title, headers, columns) in enumerate(panels):
        if p:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{title}}}}} \\")
        lines.append(" & " + " & ".join(rf"\textbf{{{h}}}" for h in headers) + r" \\")
        lines.append(r"\midrule")
        for i, (label, _) in enumerate(columns[0]):
            text = label.replace("yr⁻¹", r"yr$^{-1}$")
            text = r"\quad " + text.strip() if label.startswith("  ") else text
            cells = [c[i][1] for c in columns]
            lines.append(text + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return ("\n".join(lines).replace("%", r"\%").replace("–", "--") + "\n")


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def manuscript_table(tabular: str, report: dict, indicators: dict) -> str:
    items = report["equipment_items"]
    energy = items["vaccine_fridge_functional_by_energy_Q1109"]
    accounting = report["cohort_accounting_by_main_source"]
    no_coordinates = sum(v["no_coordinates"] for v in accounting.values())
    solar_counts = ", ".join(f"{indicators[k]['index_solar']['n']:,}" for k in list(indicators)[:-1])
    solar_counts += f" and {indicators[list(indicators)[-1]]['index_solar']['n']:,}"
    caption = (
        r"\caption{\textbf{Electricity supply and modeled disruption at hospitals, health centers and health posts.} "
        r"(A) Facility type. (B) Facilities recording selected electricity-dependent equipment as available and "
        r"functional in the SARA survey. Index: annualized electricity disruption index (yr$^{-1}$; "
        r"Equation~\ref{eq:annual_metric}), the frequency-weighted mean of event-level expected fractional supply "
        r"shortfalls. It is not an annual outage probability, an outage duration or a fraction of patients affected. "
        r"Grid-supplied facilities are evaluated with the hybrid power model and solar-supplied facilities with the "
        r"solar model; the two describe different failure mechanisms. Statistics are descriptive.}")
    notes = (
        r"Facility type from SARA question Q007 (hospitals: rural, district, general, provincial, specialized, central "
        r"and military hospitals; health centers: rural, urban and private health centers). Main electricity source "
        r"from Q410; ``No electricity reported'' is Q408 = no"
        + (f"; {WORDS[len(report['electricity_not_recorded'])]} health center has no electricity response"
           if len(report["electricity_not_recorded"]) == 1 else "") +
        r". Generator reported: a generator named as the main or alternative source (Q410, Q411); ``reported "
        r"functioning'' is the respondent's answer on the survey day (Q413). Neither establishes that a generator can "
        r"supply the facility during an outage. Evaluated: facilities with a model estimate. Not evaluated are "
        r"facilities whose main source is a generator, gas or another source, facilities without electricity, "
        f"{no_coordinates:,} facilities without coordinates and "
        f"{len(report['hybrid_unavailable_facilities']):,} grid-supplied facilities outside the modeled transmission "
        r"network; unavailable estimates are excluded, not set to zero. "
        f"{WORDS.get(len(report['solar_model_with_non_solar_main_source']), '').capitalize()} facilities whose main "
        r"source is a generator or "
        r"gas and whose alternative source is solar are evaluated with the solar model, as in Figures 3 and 4. "
        r"(B) Vaccine refrigerator: available and functional (Q1108) and powered by grid or generator electricity "
        f"(Q1109), asked of the {items['immunisation_offered_Q1100']:,} facilities offering immunization; a further "
        f"{energy.get('2.0', 0):,} functional vaccine refrigerators are solar-powered and {energy.get('3.0', 0):,} "
        r"gas-powered (not shown). Oxygen concentrator (Q500) and neonatal incubator in the delivery area (Q1005; "
        f"asked of the {items['delivery_offered_Q1000']:,} facilities offering delivery care): observed by the "
        r"enumerator and reported functional. Solar-supplied facilities recording these items "
        f"({solar_counts}) are included in ``Evaluated'' but not summarized separately. Equipment rows identify "
        r"services potentially exposed to electricity disruption; they do not predict service loss or patient harm.")
    # tabularx (loaded by the manuscript) lets the label column wrap within \linewidth.
    tabular = tabular.replace(r"\begin{tabular}{lrrrr}",
                              r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}Xrrrr}")
    tabular = tabular.replace(r"\end{tabular}", r"\end{tabularx}")
    return "\n".join([r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}", caption,
                      r"\label{tab:facility_type}", tabular.rstrip(), r"\par\smallskip",
                      r"\begin{minipage}{\linewidth}\footnotesize " + notes + r"\end{minipage}",
                      r"\end{table}"]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sara-processed", type=Path, required=True)
    parser.add_argument("--sara-raw", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--solar-per-hospital", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    f, report = load(args)
    type_summaries = {t: summarise(f[f["facility_type"] == t]) for t in TYPES}
    type_summaries["All"] = summarise(f)
    indicator_summaries = {label: summarise(f[f[column]]) for label, column in INDICATORS.items()}

    panel_a = ("A. Facility type", list(type_summaries), [rows_for(s, False) for s in type_summaries.values()])
    panel_b = ("B. Electricity-dependent equipment recorded as available and functional",
               list(indicator_summaries), [rows_for(s, True, ("index_grid",)) for s in indicator_summaries.values()])
    tables = []
    for title, headers, columns in (panel_a, panel_b):
        frame = pd.DataFrame({"": [label for label, _ in columns[0]]})
        for header, column in zip(headers, columns):
            frame[header] = [value for _, value in column]
        tables.append((title, frame))

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "facility_type_electricity_table.md").open("w") as handle:
        for title, frame in tables:
            handle.write(f"**{title}**\n\n{frame.to_markdown(index=False)}\n\n")
    pd.concat([frame.assign(panel=title) for title, frame in tables]).to_csv(
        args.output_dir / "facility_type_electricity_table.csv", index=False)
    tabular = to_latex([panel_a, panel_b])
    (args.output_dir / "facility_type_electricity_table.tex").write_text(tabular)
    (args.output_dir / "facility_type_table_manuscript.tex").write_text(
        manuscript_table(tabular, report, indicator_summaries))
    keep = ["facility_i", "node_id", "name", "region", "facility_type", "Q007", "urban", "has_coordinates",
            "main_source", "Q410", "Q411", "generator_reported", "generator_functioning", "generator_fuel_today",
            "fridge_grid", "fridge_solar", "oxygen", "incubator", "model", "grid_index", "solar_index", "index"]
    f[keep].to_csv(args.output_dir / "facility_values.csv", index=False)
    numbers = {"facility_type": type_summaries, "equipment": indicator_summaries,
               "index_definition": "mean over 2,703 events of the expected fractional shortfall x 2,703/10,000"}
    (args.output_dir / "table_numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    (args.output_dir / "join_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    for title, frame in tables:
        print(title)
        print(frame.to_string(index=False))
        print()
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
