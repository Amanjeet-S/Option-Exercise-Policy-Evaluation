"""Private scientific plots from the public-process revision experiments.

No licensed market-derived results are used. The saved tables also generate
self-contained pgfplots blocks, with all plot coordinates embedded in TeX.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["student", "empirical_bernstein", "hoeffding", "cantelli"]
COLOURS = {
    "qfblue": "#0072B2", "qfgreen": "#009E73", "qforange": "#D55E00",
    "qfamber": "#E69F00", "qfpurple": "#CC79A7", "qfviolet": "#6A3D9A",
}
NATIVE_COLOUR_DEFINITIONS = "\n".join(
    f"\\definecolor{{{name}}}{{HTML}}{{{value[1:]}}}" for name, value in COLOURS.items())+"\n"
MARGIN_CMAP = LinearSegmentedColormap.from_list(
    "qf_margin", [COLOURS["qfblue"], "#F7F7F7", COLOURS["qforange"]])
STYLES = {
    "student": ("Student", COLOURS["qfblue"], "o", "qfblue", "*"),
    "empirical_bernstein": ("Clipped empirical Bernstein", COLOURS["qfgreen"], "D", "qfgreen", "diamond*"),
    "hoeffding": ("Clipped Hoeffding", COLOURS["qfpurple"], "^", "qfpurple", "triangle*"),
    "cantelli": ("Cantelli", COLOURS["qforange"], "s", "qforange", "square*"),
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_coverage(original, clipped):
    for folder in (original, clipped):
        manifest = read(folder/"manifest.json")
        for name in ("results.json", "design.json"):
            if sha(folder/name) != manifest["artifact_sha256"][name]:
                raise ValueError("Coverage artefact checksum mismatch")
    first, second = read(original/"design.json"), read(clipped/"design.json")
    if second["original_design_sha256"] != sha(original/"design.json"):
        raise ValueError("Clipped coverage does not reference the supplied original design")
    if second["outer_counts"] != first["outer_counts"] or second["repetitions"] != first["repetitions"]:
        raise ValueError("Coverage counts differ")
    if second["cap_selection_uses_evaluation"]:
        raise ValueError("Plot requires independently selected caps")
    data = read(clipped/"results.json")
    rows = [row for row in data["records"] if row["martingale"] == "terminal_call"]
    expected = {(method, count) for method in METHODS for count in first["outer_counts"]}
    if len(rows) != len(expected) or {(row["method"], row["outer_paths"]) for row in rows} != expected:
        raise ValueError("Incomplete or duplicate centred coverage records")
    for row in rows:
        if row["repetitions"] != first["repetitions"]:
            raise ValueError("Incomplete coverage repetitions")
        interval = row["expected_gap_coverage_wilson_95"]
        if not 0 <= interval[0] <= row["expected_gap_coverage"] <= interval[1] <= 1:
            raise ValueError("Invalid coverage interval")
        if row["mean_excess_over_expected_gap"] <= 0:
            raise ValueError("Positive excess required for the logarithmic plot")
    return dict(rows=rows, outer_counts=first["outer_counts"], repetitions=first["repetitions"],
                target=data["oracle"]["true_two_right_policy_loss"], confidence=.975,
                public_process=first["process"])


def validate_matrix(folder):
    data, design, pde = read(folder/"results.json"), read(folder/"design.json"), read(folder/"one_asset_pde.json")
    if data["path_arrays_retained"] or "published public mathematical inputs" not in design["classification"]:
        raise ValueError("Require public-process aggregate simulation records without retained paths")
    expected = {(dimension, spot) for dimension in (2, 3, 5) for spot in (90., 100., 110.)}
    selected, one = [], []
    for case in data["cases"]:
        model = case["model"]
        dimension = len(model["spots"])
        spot = model["spots"][0]
        if (any(value != spot for value in model["spots"]) or model["strike"] != 100. or
                model["rate"] != .05 or model["volatilities"] != [.2]*dimension or
                model["dividends"] != [.1]*dimension or len(model["exercise_times"]) != 10 or
                not np.allclose(model["exercise_times"], np.linspace(0., 3., 10), rtol=0., atol=1e-14) or
                not np.array_equal(np.array(model["correlation"]), np.eye(dimension))):
            raise ValueError("Unexpected public-model parameters")
        if (case["design"]["evaluation_paths"] != design["evaluation_paths"] or
                case["design"]["training_paths"] != design["training_paths"]):
            raise ValueError("Matrix simulation counts differ")
        estimate = case["estimates"]["matched_gain"]
        interval = estimate["pointwise_student_interval"]
        if not interval[0] <= estimate["mean"] <= interval[1] or estimate["observations"] != design["evaluation_paths"]:
            raise ValueError("Invalid matched-gain interval")
        if dimension == 1:
            one.append(case)
        else:
            selected.append(dict(dimension=dimension, spot=spot, mean=estimate["mean"],
                                 lower=interval[0], upper=interval[1]))
    if len(one) != 1 or len(selected) != len(expected) or {(row["dimension"], row["spot"]) for row in selected} != expected:
        raise ValueError("Incomplete public-model case matrix")
    with (folder/"state_margin.csv").open() as stream:
        reader = csv.DictReader(stream)
        required = {"asset_1", "asset_2", "discounted_exercise_reward", "discounted_continuation_estimate",
                    "exercise_margin", "exercise_selected"}
        if set(reader.fieldnames) != required:
            raise ValueError("Unexpected state-margin columns")
        rows = list(reader)
    x = sorted({float(row["asset_1"]) for row in rows})
    y = sorted({float(row["asset_2"]) for row in rows})
    surface = data["state_surface"]
    if (len(rows) != len(x)*len(y) or len(rows) != surface["grid_rows"] or
            len(x) != surface["points_per_asset"] or len(y) != surface["points_per_asset"] or
            surface["time"] != surface["time_index"]/3. or
            x[0] != surface["state_grid_lower"] or y[0] != surface["state_grid_lower"] or
            x[-1] != surface["state_grid_upper"] or y[-1] != surface["state_grid_upper"]):
        raise ValueError("Incomplete state grid")
    values, locations = np.empty((len(x), len(y))), set()
    for row in rows:
        a, b, margin = float(row["asset_1"]), float(row["asset_2"]), float(row["exercise_margin"])
        reward, continuation = float(row["discounted_exercise_reward"]), float(row["discounted_continuation_estimate"])
        if (a, b) in locations or not all(math.isfinite(value) for value in (a, b, margin, reward, continuation)):
            raise ValueError("Duplicate or non-finite grid state")
        if abs(margin-(reward-continuation)) > 1e-10:
            raise ValueError("State margin does not equal exercise minus continuation")
        values[x.index(a), y.index(b)] = margin
        locations.add((a, b))
    if pde["certified_error_enclosure"] or pde["market"] != {"dividend": .1, "rate": .05, "spot": 100., "volatility": .2}:
        raise ValueError("Unexpected PDE reference or enclosure claim")
    refinements = [row for row in pde["records"] if row["category"] == "joint_refinement"]
    if len(refinements) < 3 or any(row["time_steps"] % 9 for row in refinements):
        raise ValueError("Require exercise-aligned PDE refinement")
    return dict(gains=selected, one_asset=one[0]["estimates"]["policy_value"], pde=pde,
                refinements=refinements, x=x, y=y, margin=values.tolist(), surface=surface, design=design)


def save_figure(fig, output, stem, caption, artefacts):
    fig.savefig(output/f"{stem}.pdf", bbox_inches="tight", metadata={"Author": "Amanjeet Singh", "Title": caption,
                "Creator": "Amanjeet Singh; Matplotlib"})
    fig.savefig(output/f"{stem}.png", dpi=150, bbox_inches="tight", metadata={"Author": "Amanjeet Singh", "Description": caption})
    plt.close(fig)
    artefacts.append(dict(stem=stem, pdf=f"{stem}.pdf", png=f"{stem}.png", caption=caption))


def pgf_table(rows, names):
    return " ".join(names)+"\n"+"\n".join(" ".join(f"{value:.17g}" for value in row) for row in rows)+"\n"


def pgf_plot(rows, colour, label, *, errors=False, marker="*"):
    options = f"color={colour},mark={marker},thick"
    if errors:
        options += ",error bars/.cd,y dir=both,y explicit"
        table_options, columns = "x=x,y=y,y error plus=hi,y error minus=lo", ["x", "y", "hi", "lo"]
    else:
        table_options, columns = "x=x,y=y", ["x", "y"]
    return (f"\\addplot+[{options}] table[{table_options}] {{\n{pgf_table(rows, columns)}}};\n"
            +f"\\addlegendentry{{{label}}}\n")


def figure_block(axis, plots, caption, label):
    return ("\\begin{figure}[htbp]\n\\centering\n\\begin{tikzpicture}\n"
            +f"\\begin{{axis}}[{axis}]\n"+plots+"\\end{axis}\n\\end{tikzpicture}\n"
            +f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n\n")


def make_figures(coverage, matrix, output):
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 8.5,
                        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    artefacts, blocks = [], []
    common_pgf = ("width=.93\\linewidth,height=6cm,xlabel={Outer paths},xmode=log,log basis x=2,"
                  "xtick={8,32,128,512,2048},xticklabels={8,32,128,512,2048},grid=major,"
                  "legend columns=2,legend style={at={(0.5,-0.30)},anchor=north,font=\\footnotesize,draw=none,cells={anchor=west}}")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    plots = ""
    for method in METHODS:
        rows = sorted((row for row in coverage["rows"] if row["method"] == method), key=lambda row: row["outer_paths"])
        label, colour, marker, texcolour, texmarker = STYLES[method]
        x, value = np.array([row["outer_paths"] for row in rows]), np.array([row["expected_gap_coverage"] for row in rows])
        lo, hi = np.array([row["expected_gap_coverage_wilson_95"][0] for row in rows]), np.array([row["expected_gap_coverage_wilson_95"][1] for row in rows])
        ax.errorbar(x, value, yerr=[value-lo, hi-value], color=colour, marker=marker, linewidth=1.4, capsize=3, label=label)
        plots += pgf_plot(list(zip(x, value, hi-value, value-lo)), texcolour, label,
                          errors=True, marker=texmarker)
    ax.axhline(.975, color="0.3", linestyle="--", linewidth=1., label="Nominal 97.5%")
    ax.set(xscale="log", xlabel="Outer paths", ylabel="One-sided coverage\n(per cent)", ylim=(.69, 1.015))
    ax.set_xticks(coverage["outer_counts"], labels=[str(count) for count in coverage["outer_counts"]])
    ax.yaxis.set_major_formatter(PercentFormatter(1.))
    ax.grid(alpha=.2); ax.legend(loc="upper center", bbox_to_anchor=(.5,-.25), ncol=2, frameon=False)
    caption = ("Coverage for the exactly centred terminal-call martingale and the always-wait policy, with 5,000 independent repetitions per path count. Bars are Wilson 95% intervals. Clipping caps are selected on an independent pilot; coverage is measured against the disclosed numerical policy-loss target.")
    save_figure(fig, output, "coverage", caption, artefacts)
    plots += "\\addplot[black,dashed,mark=none] coordinates {(8,.975) (2048,.975)};\\addlegendentry{Nominal 97.5\\%}\n"
    blocks.append(figure_block(common_pgf+",ylabel={Observed coverage (per cent)},ymin=.69,ymax=1.015,"
        "ytick={.70,.75,.80,.85,.90,.95,1.00},yticklabels={70,75,80,85,90,95,100}", plots,
        caption.replace("95%", "95\\%"), "fig:revision-coverage"))

    fig, ax = plt.subplots(figsize=(6.5, 3.8)); plots = ""
    for method in METHODS:
        rows = sorted((row for row in coverage["rows"] if row["method"] == method), key=lambda row: row["outer_paths"])
        label, colour, marker, texcolour, texmarker = STYLES[method]
        x, excess = [row["outer_paths"] for row in rows], [row["mean_excess_over_expected_gap"] for row in rows]
        ax.plot(x, excess, color=colour, marker=marker, linewidth=1.4, label=label)
        plots += pgf_plot(list(zip(x, excess)), texcolour, label, marker=texmarker)
    ax.set(xscale="log", yscale="log", xlabel="Outer paths", ylabel="Mean upper-limit excess (price units)")
    ax.set_xticks(coverage["outer_counts"], labels=[str(count) for count in coverage["outer_counts"]])
    ax.grid(alpha=.2); ax.legend(loc="upper center", bbox_to_anchor=(.5,-.25), ncol=2, frameon=False)
    caption = (f"Mean upper-limit excess above the two-right policy-loss target {coverage['target']:.6f}, under the same simulation design. Both axes are logarithmic. The ordinate measures excess above the target, rather than an interval width.")
    save_figure(fig, output, "upper_limit_excess", caption, artefacts)
    blocks.append(figure_block(common_pgf+",ymode=log,ylabel={Mean upper-limit excess (price units)}", plots, caption,
                               "fig:revision-upper-excess"))

    fig, ax = plt.subplots(figsize=(6.5, 3.8)); plots = ""
    for spot, colour, marker, texcolour, texmarker in [
            (90., COLOURS["qfblue"], "o", "qfblue", "*"),
            (100., COLOURS["qfgreen"], "D", "qfgreen", "diamond*"),
            (110., COLOURS["qforange"], "s", "qforange", "square*")]:
        rows = sorted((row for row in matrix["gains"] if row["spot"] == spot), key=lambda row: row["dimension"])
        x, means = np.array([row["dimension"] for row in rows]), np.array([row["mean"] for row in rows])
        lower, upper = np.array([row["lower"] for row in rows]), np.array([row["upper"] for row in rows])
        ax.errorbar(x, means, yerr=[means-lower, upper-means], color=colour, marker=marker, capsize=4, label=f"Initial spot {spot:g}")
        plots += pgf_plot(list(zip(x, means, upper-means, means-lower)), texcolour,
                          f"Initial spot {spot:g}", errors=True, marker=texmarker)
    ax.set(xlabel="Asset dimension", ylabel="Gain over terminal waiting\n(price units)", xticks=[2, 3, 5]); ax.grid(alpha=.2); ax.legend(loc="upper center", bbox_to_anchor=(.5,-.25), ncol=3, frameon=False)
    caption = ("Matched exercise-policy gain for the public max-call process. Each degree-2 full-state polynomial policy uses 50,000 training paths and 200,000 independent evaluation paths. Bars are approximate pointwise 95% Student intervals conditional on the fitted policy. Cases share a seed layout; the intervals make no simultaneous or independent-case claim.")
    save_figure(fig, output, "dimension_gains", caption, artefacts)
    blocks.append(figure_block("width=.93\\linewidth,height=6cm,xlabel={Asset dimension},ylabel={Gain over terminal waiting\n(price units)},xtick={2,3,5},grid=major,legend columns=3,legend style={at={(0.5,-0.30)},anchor=north,font=\\footnotesize,draw=none}", plots,
                               caption.replace("95%", "95\\%"), "fig:revision-dimension-gains"))

    x, y, z = np.array(matrix["x"]), np.array(matrix["y"]), np.array(matrix["margin"])
    span = max(abs(z.min()), abs(z.max()))
    normalise = TwoSlopeNorm(vmin=-span, vcenter=0., vmax=span)
    fig, ax = plt.subplots(figsize=(6., 4.5))
    heat = ax.pcolormesh(x, y, z.T, shading="auto", cmap=MARGIN_CMAP, norm=normalise)
    contour = ax.contour(x, y, z.T, levels=[0.], colors="black", linewidths=.9)
    contour_segments = [segment.tolist() for segment in contour.allsegs[0] if len(segment) > 1]
    ax.set(xlabel="Asset 1 state (price units)", ylabel="Asset 2 state (price units)", aspect="equal")
    fig.colorbar(heat, ax=ax, label="Exercise margin (time-zero price units)")
    caption = (f"Fitted two-asset max-call policy margin at t={matrix['surface']['time']:g} years, in time-zero price units. Positive margins favour exercise; ties continue. The full saved 41 by 41 state grid and its zero contour are shown. This is a fitted-policy diagnostic, not an optimal exercise boundary; polynomial extrapolation limits apply.")
    save_figure(fig, output, "state_margin_heatmap", caption, artefacts)
    native_grid = [[float(x[i]), float(y[j]), float(z[i, j])]
                   for j in range(len(y)) for i in range(len(x))]
    matrix["pgf_heatmap_grid"] = native_grid
    matrix["zero_contour_segments"] = contour_segments
    colormap = ("colormap={qfdiv}{rgb255(0cm)=(0,114,178);"
                "rgb255(1cm)=(247,247,247);rgb255(2cm)=(213,94,0)},"
                +f"point meta min={-span:.17g},point meta max={span:.17g},colorbar,"
                "colorbar style={ylabel={Exercise margin (time-zero price units)},"
                "ylabel style={font=\\small},tick label style={font=\\small}}")
    heatmap_plot = (f"\\addplot[matrix plot*,mesh/cols={len(x)},point meta=explicit] "
                   "table[x=x,y=y,meta=z] {\n"+pgf_table(native_grid, ["x", "y", "z"])+"};\n")
    for segment in contour_segments:
        coordinates = " ".join(f"({a:.17g},{b:.17g})" for a, b in segment)
        heatmap_plot += "\\addplot[black,thick,no marks] coordinates {"+coordinates+"};\n"
    blocks.append(figure_block("width=.78\\linewidth,height=7cm,axis equal image,"
        "xlabel={Asset 1 state (price units)},ylabel={Asset 2 state (price units)},"+colormap,
        heatmap_plot, caption, "fig:revision-state-margin"))

    reference = matrix["pde"]["finest_time_value"]
    points = [[row["space_steps"], abs(row["value"]-reference)] for row in matrix["refinements"]]
    estimate = matrix["one_asset"]; low, high = estimate["pointwise_student_interval"]
    fig, axes = plt.subplots(1, 2, figsize=(7., 3.5))
    axes[0].plot([row[0] for row in points], [row[1] for row in points], "o-", color=COLOURS["qfblue"], label="PDE refinement")
    axes[0].set(xscale="log", yscale="log", xlabel="Spatial intervals", ylabel="PDE difference\n(price units)")
    axes[0].set_xticks([row[0] for row in points], labels=[str(row[0]) for row in points])
    axes[0].minorticks_off()
    axes[0].grid(alpha=.2)
    axes[1].errorbar([0], [estimate["mean"]], yerr=[[estimate["mean"]-low], [high-estimate["mean"]]], color=COLOURS["qfblue"], marker="o", capsize=5, label="LSM")
    axes[1].plot([1], [reference], "D", color=COLOURS["qforange"], label="PDE point reference")
    axes[1].set(xticks=[0, 1], xticklabels=["LSM", "PDE"], ylabel="Option value\n(price units)", xlim=(-.5, 1.5)); axes[1].grid(axis="y", alpha=.2)
    for panel in axes:
        panel.legend(loc="upper center", bbox_to_anchor=(.5,-.26), frameon=False)
    fig.tight_layout()
    caption = ("One-asset public-model reference. Left: joint spatial/time refinement compared with the finer-time PDE point reference. Right: the fitted policy value and its approximate pointwise 95% sampling interval beside that PDE value. These numerical convergence checks are not certified price or loss enclosures.")
    save_figure(fig, output, "pde_reference", caption, artefacts)
    left_axis = ("width=.43\\linewidth,height=5.8cm,xmode=log,ymode=log,xlabel={Spatial intervals},"
                 "ylabel={PDE difference\n(price units)},xtick={400,800,1600,3200},"
                 "xticklabels={400,800,1600,3200},grid=major,legend columns=1,legend style={at={(0.5,-0.35)},anchor=north,font=\\footnotesize,draw=none}")
    right_axis = ("width=.40\\linewidth,height=5.8cm,xlabel={},ylabel={Option value\n(price units)},"
                  "xmin=-.5,xmax=1.5,xtick={0,1},xticklabels={LSM,PDE},grid=major,legend columns=1,legend style={at={(0.5,-0.35)},anchor=north,font=\\footnotesize,draw=none}")
    left_plot = pgf_plot(points, "qfblue", "PDE refinement")
    right_plot = pgf_plot([[0., estimate["mean"], high-estimate["mean"], estimate["mean"]-low]],
                         "qfblue", "LSM", errors=True)
    right_plot += pgf_plot([[1., reference]], "qforange", "PDE point reference", marker="diamond*")
    blocks.append("\\begin{figure}[htbp]\n\\centering\n\\begin{tikzpicture}\n"
        +f"\\begin{{axis}}[{left_axis}]\n"+left_plot+"\\end{axis}\n"
        +f"\\begin{{axis}}[at={{(7cm,0)}},anchor=south west,{right_axis}]\n"
        +right_plot+"\\end{axis}\n\\end{tikzpicture}\n"
        +f"\\caption{{{caption.replace('95%', '95' + chr(92) + '%')}}}\n"
        +"\\label{fig:revision-pde-reference}\n\\end{figure}\n\n")
    matrix["pde_plot"] = dict(refinement_differences=points, finer_time_reference=reference,
                              lsm_mean=estimate["mean"], lsm_interval=[low, high])
    return artefacts, blocks



VERIFIED_STYLES = {
    'hinge/cold_uniform': ('Uniform hinge', COLOURS['qfblue'], 's', 'qfblue', 'square*', ''),
    'hinge/adaptive': ('Adaptive hinge', COLOURS['qfgreen'], 'o', 'qfgreen', '*', ''),
    '2/polynomial_all_dates': ('Polynomial, degree 2', COLOURS['qforange'], '^', 'qforange', 'triangle*', ''),
    '2/hybrid_exact_terminal': ('Hybrid, degree 2', COLOURS['qfamber'], 'D', 'qfamber', 'diamond*', ''),
    '3/polynomial_all_dates': ('Polynomial, degree 3', COLOURS['qfpurple'], 'v', 'qfpurple', 'triangle*', ',mark options={rotate=180}'),
    '3/hybrid_exact_terminal': ('Hybrid, degree 3', COLOURS['qfviolet'], 'p', 'qfviolet', 'pentagon*', ''),
}


def verified_frontier_rows(study):
    """Obtain six checked cost/quality coordinates without loading path arrays.

    Each gap standard error is calculated across eight fitting-unit means,
    not by treating the 2,048 outer paths as independent fitted models. CPU
    ranges are descriptive minimum/maximum observations, not confidence
    intervals. The same frozen policy is used within each fitting unit.
    """
    from importlib.util import module_from_spec, spec_from_file_location
    specification = spec_from_file_location('revision_plot_checks', ROOT/'paper/verification/revision_results.py')
    checks = module_from_spec(specification)
    specification.loader.exec_module(checks)
    checks.validate_replicated_refinement(study)
    rows = []
    for method, (label, colour, marker, texcolour, texmarker, texoptions) in VERIFIED_STYLES.items():
        gap = study['gap_midpoint_by_method'][method]
        cost = study['standalone_verified_budget_cpu_seconds'][method]
        if gap['fitting_units'] != 8 or cost['fitting_units'] != 8:
            raise ValueError('Cost/quality coordinates require eight complete fitting units')
        rows.append(dict(method=method, label=label, colour=colour, marker=marker,
            texcolour=texcolour, texmarker=texmarker, texoptions=texoptions,
            cpu_median=cost['median'], cpu_minimum=cost['minimum'], cpu_maximum=cost['maximum'],
            gap_mean=gap['mean'], gap_standard_error=gap['standard_error']))
    return rows


def verified_cost_quality_block(rows):
    """Use the same checked cost/gap coordinates and encodings in native TeX."""
    plots = ''
    for row in rows:
        values = [[row['cpu_median'], row['gap_mean'],
                   row['cpu_median']-row['cpu_minimum'],
                   row['cpu_maximum']-row['cpu_median'], row['gap_standard_error']]]
        options = (f"only marks,mark={row['texmarker']},mark size=2.8pt,color={row['texcolour']}"
                   +row['texoptions']+",error bars/.cd,x dir=both,x explicit,y dir=both,y explicit,"
                   "error bar style={line width=0.55pt}")
        plots += (f"\\addplot+[{options}] table[x=cpu,y=gap,x error minus=cpu_minus,"
                  "x error plus=cpu_plus,y error=se] {\n"
                  +pgf_table(values, ['cpu','gap','cpu_minus','cpu_plus','se'])+"};\n"
                  +f"\\addlegendentry{{{row['label']}}}\n")
    caption = ("Cost and quality of verified dual assessments across eight independent fits. "
               "Points show the median preparation and assessment budget and mean stopped gap. Horizontal bars "
               "span observed minimum and maximum CPU; vertical bars give one across-fit standard "
               "error of the gap mean. These ranges and standard errors are not confidence intervals. "
               "Budgets include policy validation. Process start-up and result serialisation are excluded. "
               "Every method attains maximum "
               "path-gap width 0.01.")
    axis = ("width=.97\\textwidth,height=.53\\textwidth,"
            "xlabel={Preparation and assessment CPU (seconds)},ylabel={Mean stopped gap (price units)},"
            "xmin=0,ymin=0,grid=major,grid style={gray!20},tick label style={font=\\small},"
            "label style={font=\\small},legend columns=2,legend style={at={(0.5,-0.30)},anchor=north,"
            "draw=none,font=\\footnotesize,cells={anchor=west}},"
            "enlarge x limits={upper,value=.05},enlarge y limits={upper,value=.07}")
    return figure_block(axis, plots, caption, 'fig:revision-replicated-tradeoff')


def make_verified_cost_quality(study, output):
    """Save only the new PDF; existing revision figures remain unchanged."""
    output = Path(output)
    if output.suffix.lower() != '.pdf' or output.exists():
        raise ValueError('Use a new PDF file for the verified cost/quality figure')
    rows = verified_frontier_rows(study)
    with plt.rc_context({'font.size':10, 'axes.labelsize':11, 'pdf.fonttype':42, 'ps.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False}):
        fig, ax = plt.subplots(figsize=(7.3, 5.8))
        fig.subplots_adjust(left=.11, right=.975, bottom=.40, top=.88)
        maximum_cost = max(row['cpu_maximum'] for row in rows)
        maximum_gap = max(row['gap_mean']+row['gap_standard_error'] for row in rows)
        for row in rows:
            x, y = row['cpu_median'], row['gap_mean']
            ax.errorbar(x, y,
                xerr=[[x-row['cpu_minimum']], [row['cpu_maximum']-x]],
                yerr=row['gap_standard_error'], fmt=row['marker'], color=row['colour'],
                markersize=7.3, markeredgecolor='white', markeredgewidth=.7,
                elinewidth=1.25, capsize=3, capthick=1.15, zorder=3, label=row['label'])
        ax.set(xlim=(0., maximum_cost*1.08), ylim=(0., maximum_gap*1.13),
               xlabel='Preparation and assessment CPU (seconds)',
               ylabel='Mean stopped gap (price units)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1,2,2.5,5,10]))
        ax.grid(alpha=.17, linewidth=.7)
        ax.legend(loc='upper center', bbox_to_anchor=(.5,-.22), ncol=2, frameon=False, fontsize=9)
        ax.set_title('Verified cost and dual-gap quality', fontsize=12, loc='left', pad=11)
        fig.text(.975,.975,'Maximum path-gap width: 0.01; five assets; eight fitted models',
                 ha='right',va='top',fontsize=8.5,color='0.3')
        fig.text(.11, .09, 'Points: median preparation and assessment CPU; horizontal bars: observed fit range.',
                 fontsize=8.5, ha='left')
        fig.text(.11, .055, 'Vertical bars: mean gap plus or minus one standard error across eight fitted models.',
                 fontsize=8.5, ha='left')
        fig.text(.11, .02, 'Budgets include policy validation; process start-up and result serialisation are excluded.',
                 fontsize=8.5, ha='left')
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, metadata={
            'Author':'Amanjeet Singh', 'Title':'Verified cost and dual-gap quality',
            'Creator':'Amanjeet Singh; Matplotlib',
            'Subject':'Six methods, eight independently fitted five-asset models, 256 outer paths per fit; medians of preparation and assessment budgets include policy validation but exclude process start-up and result serialisation. Horizontal CPU ranges are descriptive and vertical bars show one across-fit standard error.',
            'Keywords':'optimal stopping; martingale duality; verified numerical integration; computational allocation',
        })
        plt.close(fig)
    return rows


def run_verified_cost_quality(args):
    input_path = args.revision_results.resolve()
    before = sha(input_path)
    study = read(input_path)['replicated_refinement']
    output = args.output.resolve()
    rows = make_verified_cost_quality(study, output)
    if sha(input_path) != before:
        raise ValueError('The retained replicated results changed during figure generation')
    print('Created the verified cost/quality PDF without rewriting existing figures.', flush=True)
    return rows

def run(args):
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/"data/private").resolve()):
        raise ValueError("Keep plots private")
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    source_paths = [args.coverage_v1/"results.json", args.coverage_v1/"design.json", args.coverage_v2/"results.json",
                    args.coverage_v2/"design.json", args.matrix/"results.json", args.matrix/"design.json",
                    args.matrix/"state_margin.csv", args.matrix/"one_asset_pde.json"]
    inputs = {str(path.resolve()): sha(path) for path in source_paths}
    coverage, matrix = validate_coverage(args.coverage_v1, args.coverage_v2), validate_matrix(args.matrix)
    output.mkdir(parents=True)
    artefacts, blocks = make_figures(coverage, matrix, output)
    tex = ("% Self-contained plot data. Required preamble packages: tikz, pgfplots.\n"
        "% Figures use only stipulated/public-process simulation results.\n\\pgfplotsset{compat=1.18}\n"
        +NATIVE_COLOUR_DEFINITIONS+"\n"+"".join(blocks))
    (output/"figure_blocks.tex").write_text(tex)
    data = dict(coverage=coverage, matrix=matrix, author="Amanjeet Singh", input_sha256=inputs,
                artefacts=artefacts, contains_subscription_numbers=False, retained_path_arrays=False)
    (output/"plot_data.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    for path in source_paths:
        if sha(path) != inputs[str(path.resolve())]:
            raise ValueError("Plot input changed during rendering")
    names = [path.name for path in output.iterdir() if path.is_file()]
    size = sum((output/name).stat().st_size for name in names)
    if size > 3*1024**2:
        raise ValueError("Figure deliverables exceed the requested 3 MiB limit")
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), author="Amanjeet Singh",
        input_sha256=inputs, source_sha256=sha(Path(__file__)), total_bytes_before_manifest=size,
        artifact_sha256={name: sha(output/name) for name in names}, figures=artefacts,
        native_heatmap_grid=[41, 41], matplotlib_heatmap_grid=[41, 41],
        native_heatmap_points=1681, zero_contour_segments=len(matrix['zero_contour_segments']),
        native_contour_source='same full-grid Matplotlib zero-contour calculation',
        publication_clearance=False)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(f"Created {len(artefacts)} private PDF/PNG figures; total deliverables {size/1024:.1f} KiB", flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--coverage-v1", type=Path)
    result.add_argument("--coverage-v2", type=Path)
    result.add_argument("--matrix", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--verified-cost-quality-only", action="store_true",
                        help="Create only the new six-method verified cost/quality PDF")
    result.add_argument("--revision-results", type=Path, default=ROOT/"paper/revision_results.json")
    return result


if __name__ == "__main__":
    command = parser()
    arguments = command.parse_args()
    if arguments.verified_cost_quality_only:
        run_verified_cost_quality(arguments)
    else:
        missing = [name for name in ("coverage_v1", "coverage_v2", "matrix") if getattr(arguments,name) is None]
        if missing:
            command.error("Original figure generation requires "+", ".join("--"+name.replace("_","-") for name in missing))
        run(arguments)
