"""The paper's tables as LaTeX, each cell the mean +- SD over seeds 42/43/44.

Every number comes from the results JSONs ``run_metacog_grid.sh`` writes, one per dataset and seed:
AU-PRO and image AUROC, and the F1 at each threshold rule as ``f1_<rule>``.

The SD is the spread over three coreset draws and three brightness-augmentation streams. The
backbone is frozen and the Mahalanobis fit takes no seed, so it measures noise in the fit, not in
the data or the encoder. With n = 3 it is itself noisy, and it is not a confidence interval.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parent / "out"
SEEDS = tuple(int(s) for s in os.environ.get("METACOG_SEEDS", "42,43,44").split(","))
assert os.environ.get("METACOG_RUN_DIR"), (
    "set METACOG_RUN_DIR to the directory run_metacog_grid.sh wrote (results/<ds>/seed<N>/<ds>.json)"
)
RUN = pathlib.Path(os.environ["METACOG_RUN_DIR"])

ARMS = ("knn", "mahalanobis")
ARM_HEADER = "kNN & Mahalanobis"
#: (key, label, the AU-PRO limit its per-category table prints)
DATASETS = (("ad", "MVTec AD", "au_pro@0.3"),
            ("ad2", "MVTec AD~2", "au_pro@0.05"),
            ("visa", "VisA", "au_pro@0.3"))
RULE_LABELS = {"q95": r"$Q_{95}$", "q99": r"$Q_{99}$", "mu3sigma": r"$\mu{+}3\sigma$",
               "evt": r"EVT $\gamma_c$", "f1max": r"$F_1^{\max}$"}
#: The rules in the per-category tables, and in the summary. The summary leaves out Q95: as a rule
#: it is a 5% per-pixel false-positive rate, and its F1 is near zero everywhere.
CATEGORY_RULES = ("q95", "q99", "mu3sigma", "evt", "f1max")
SUMMARY_RULES = ("q99", "mu3sigma", "evt", "f1max")

_CACHE: dict[str, dict] = {}


def results(ds: str) -> list[dict]:
    """The results JSON of every seed that exists, in seed order."""
    paths = [RUN / "results" / ds / f"seed{seed}" / f"{ds}.json" for seed in SEEDS]
    return [json.loads(path.read_text()) for path in paths if path.exists()]


def load(ds: str) -> dict:
    """``{category: {arm: {metric: [value per seed]}}}``, with F1 as a percentage."""
    if ds not in _CACHE:
        out: dict = {}
        for result in results(ds):
            for category, entry in result["categories"].items():
                for arm, arm_result in entry.get("arms", {}).items():
                    for key, value in arm_result["metrics"].items():
                        scale = 100.0 if key.startswith("f1_") else 1.0
                        out.setdefault(category, {}).setdefault(arm, {}).setdefault(key, []).append(
                            float(value) * scale)
        _CACHE[ds] = out
    return _CACHE[ds]


def values(data: dict, category: str, arm: str, metric: str) -> list:
    return data.get(category, {}).get(arm, {}).get(metric, [])


def dataset_mean(data: dict, arm: str, metric: str) -> list:
    """The mean over categories, once per seed, so the SD is over seeds and not over categories."""
    series = [values(data, c, arm, metric) for c in sorted(data)]
    series = [s for s in series if s]
    if not series:
        return []
    return [float(np.mean([s[i] for s in series])) for i in range(min(len(s) for s in series))]


def cell(samples, digits: int) -> str:
    """``mean +- SD``, or the bare mean for a single seed.

    A nonzero SD that would round to zero is printed as ``<`` the smallest printable value, not as
    ``+-0``: the Mahalanobis arm's SD is 1e-6 to 4e-5 in several cells, which is small, not zero.
    """
    a = np.asarray(samples, dtype=float)
    if a.size == 0:
        return "---"
    if a.size == 1:
        return f"{a[0]:.{digits}f}"
    mean, sd = a.mean(), a.std(ddof=1)
    smallest = 10.0 ** -digits
    if sd == 0.0:
        tail = r"$\pm$0"
    elif sd < smallest:
        tail = f"$<${smallest:.{digits}f}"
    else:
        tail = f"$\\pm${sd:.{digits}f}"
    return f"{mean:.{digits}f}\\,\\tiny{tail}"


def bold_pair(cells: list) -> list:
    """Bold the larger mean of one (kNN, Mahalanobis) pair. The SD is not consulted."""
    try:
        a, b = (float(c.split("\\,")[0]) for c in cells)
    except ValueError:
        return cells
    if a == b:
        return cells
    out = list(cells)
    win = 0 if a > b else 1
    out[win] = r"\textbf{" + out[win] + "}"
    return out


def _tex(category: str) -> str:
    return r"\texttt{" + category.replace("_", r"\_") + "}"


def _pairs_header(labels) -> tuple[str, str]:
    """The ``\\multicolumn`` header and ``\\cmidrule`` line over (kNN, Mahalanobis) column pairs."""
    header = " & ".join(r"\multicolumn{2}{c}{" + label + "}" for label in labels)
    rules = "".join(r"\cmidrule(lr){" + f"{2 + 2 * i}-{3 + 2 * i}" + "}" for i in range(len(labels)))
    return header, rules


def _table(env: str, columns: str, labels, first: str, rows: list, caption: str, label: str = "",
           sep: int = 4) -> str:
    header, rules = _pairs_header(labels)
    return "\n".join([
        r"\begin{" + env + "}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{" + f"{sep}pt" + "}",
        r"\begin{tabular}{" + columns + "}", r"\toprule",
        "& " + header + r" \\", rules,
        first + " & " + " & ".join([ARM_HEADER] * len(labels)) + r" \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{" + caption + "}", *([r"\label{" + label + "}"] if label else []),
        r"\end{" + env + "}",
    ])


SPREAD = ("A cell with a $\\pm$ is the mean over seeds " + "/".join(map(str, SEEDS)) + " $\\pm$ the "
          "sample standard deviation. The seed drives the coreset subsample and the brightness-"
          "augmentation stream; the backbone is frozen and the Mahalanobis fit takes no seed, so the "
          "spread is noise in the fit, not a confidence interval.")
RULES_NOTE = ("The rules left of the vertical line are derived from held-out normal pixels alone, "
              "without reading a test mask. $F_1^{\\max}$ is the best F1 over the gain sweep chosen "
              "\\emph{with the test masks}: a ceiling, not a rule. Native resolution, exact ground "
              "truth, the deployed morphology scaled to native size, and every test image.")


def table_aupro(ds: str, label: str, metric: str) -> str:
    data = load(ds)
    if not data:
        return ""
    rows = [f"{_tex(c)} & " + " & ".join(cell(values(data, c, arm, m), 4)
                                         for m in (metric, "image_auroc") for arm in ARMS) + r" \\"
            for c in sorted(data)]
    rows += [r"\midrule", "mean & " + " & ".join(cell(dataset_mean(data, arm, m), 4)
                                                  for m in (metric, "image_auroc") for arm in ARMS) + r" \\"]
    head = metric.replace("au_pro@", "AU-PRO$_{") + "}$"
    return _table("table", "@{}lcc@{\\hspace{1.3em}}cc@{}", [head, "image AUROC"], "category", rows,
                  f"{label}, ViT-H+: {head} and image AUROC per category. {SPREAD}", sep=5)


def table_rules(ds: str, label: str) -> str:
    data = load(ds)
    if not any(k.startswith("f1_") for arms in data.values() for m in arms.values() for k in m):
        return ""
    keys = [f"f1_{rule}" for rule in CATEGORY_RULES]
    rows = [f"{_tex(c)} & " + " & ".join(cell(values(data, c, arm, k), 2)
                                         for k in keys for arm in ARMS) + r" \\"
            for c in sorted(data)]
    rows += [r"\midrule", "mean & " + " & ".join(cell(dataset_mean(data, arm, k), 2)
                                                  for k in keys for arm in ARMS) + r" \\"]
    columns = "@{}l" + "cc" * (len(keys) - 1) + "|cc@{}"
    return _table("table*", columns, [RULE_LABELS[r] for r in CATEGORY_RULES], "category", rows,
                  f"{label}, ViT-H+: segmentation $F_1$ (\\%) per category at each threshold rule. "
                  f"{SPREAD} {RULES_NOTE}")


def table_summary_free() -> str:
    """Threshold-free dataset means, in two blocks by the FPR limit the benchmark integrates to.

    The blocks are not comparable down a column: the limit changes the ordering of the two scorers
    on MVTec AD, where the kNN arm leads at 0.3 and loses at 0.05.
    """
    blocks = ((r"$\aupro{0.3}$", "au_pro@0.3", ("ad", "visa")),
              (r"$\aupro{0.05}$", "au_pro@0.05", ("ad2",)))
    rows = []
    for i, (title, metric, keys) in enumerate(blocks):
        if i:
            rows.append(r"\noalign{\vskip 2pt}\midrule")
        rows.append(r"\multicolumn{5}{@{}l}{" + title + r"} \\")
        for ds, label, _ in DATASETS:
            if ds in keys:
                data = load(ds)
                cells = [c for m in (metric, "image_auroc")
                         for c in bold_pair([cell(dataset_mean(data, arm, m), 4) for arm in ARMS])]
                rows.append(f"\\quad {label} & " + " & ".join(cells) + r" \\")
    return _table("table", "@{}lcc@{\\hspace{1.3em}}cc@{}", ["AU-PRO", "image AUROC"], "dataset", rows,
                  "Threshold-free monitoring: dataset means over all categories. " + SPREAD +
                  r" \textbf{The two blocks are not comparable}: AU-PRO is integrated to the "
                  "false-positive rate each benchmark prescribes, $0.05$ for MVTec AD~2 and $0.3$ for "
                  "MVTec AD and VisA, and on MVTec AD the ordering of the two scorers reverses between "
                  "the two limits. Image AUROC may be read down the column.",
                  "tab:summary_free", sep=5)


def table_summary_rules() -> str:
    rows = []
    for ds, label, _ in DATASETS:
        data = load(ds)
        cells = [c for rule in SUMMARY_RULES
                 for c in bold_pair([cell(dataset_mean(data, arm, f"f1_{rule}"), 2) for arm in ARMS])]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    columns = "@{}l" + "cc" * (len(SUMMARY_RULES) - 1) + "|cc@{}"
    return _table("table", columns, [RULE_LABELS[r] for r in SUMMARY_RULES], "dataset", rows,
                  "Segmentation $F_1$ (\\%) at a committed operating point: dataset means. " + SPREAD +
                  " " + RULES_NOTE + " $Q_{95}$ is left out: as a rule it is a 5\\% per-pixel "
                  "false-positive rate, and its $F_1$ is near zero on every dataset.",
                  "tab:summary_rules")


def pooled_fpr(ds: str, arm: str, k: int) -> list:
    """False-positive pixels per 640 x 640 normal image, pooled over every category, once per seed."""
    out = []
    for result in results(ds):
        records = [entry["arms"][arm]["evt_fpr_cv"][str(k)] for entry in result["categories"].values()
                   if "evt_fpr_cv" in entry.get("arms", {}).get(arm, {})]
        if records:
            out.append(sum(r["exceed"] for r in records) / sum(r["pixels"] for r in records) * 640 * 640)
    return out


def table_fpr_cv() -> str:
    """The cross-validation behind k = 4: the false-positive pixels per 640 x 640 normal image the EVT
    threshold delivers on held-out normal images it was not fitted on, for k = 3, 4 and 5."""
    from scipy.stats import norm

    rows = []
    for ds, label, _ in DATASETS:
        for k in (3, 4, 5):
            cells = [cell(pooled_fpr(ds, arm, k), 2) for arm in ARMS]
            rows.append(f"{label if k == 3 else ''} & {k} & {norm.sf(k) * 640 * 640:.2f} & "
                        + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    if all("---" in row for row in rows if "&" in row):
        return ""
    rows[-1] = r"\bottomrule"
    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
        r"dataset & $k$ & target & kNN & Mahalanobis \\", r"\midrule", *rows,
        r"\end{tabular}",
        r"\caption{False-positive pixels per $640 \times 640$ normal image at the EVT threshold "
        r"$\hat q_{1-\alpha_k}$, $\alpha_k = 1 - \Phi(k)$. The target is $\alpha_k$ times the pixel "
        r"count. The delivered count is measured by 5-fold cross-validation over the held-out normal "
        r"images and pooled over every category. " + SPREAD + "}",
        r"\label{tab:fpr_cv}", r"\end{table}",
    ])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    outputs = [("seeds_summary_free", table_summary_free()),
               ("seeds_summary_rules", table_summary_rules()),
               ("fpr_cv", table_fpr_cv())]
    for ds, label, metric in DATASETS:
        outputs += [(f"seeds_aupro_{ds}", table_aupro(ds, label, metric)),
                    (f"seeds_rules_{ds}", table_rules(ds, label))]
    for name, body in outputs:
        if not body:
            print(f"skip {name}: no results for this dataset")
            continue
        path = OUT / f"{name}.tex"
        path.write_text("% Generated by tables/tables_seeds.py -- do not hand-edit.\n" + body + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
