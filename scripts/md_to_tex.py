"""把本项目的 Markdown 报告转换为可编译的 NeurIPS LaTeX 稿件。

只处理本项目实际使用的语法子集：标题、段落、粗斜体、行内代码、表格、
图片、行间公式、引用标记。不追求通用。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

TITLE = "Calibration Is Not Enough: A Three-Level Audit of Selective Prediction for Multi-Label ECG Classification"
AUTHOR = r"""Zhengyu \\
MD6329 Deep Learning, Nanyang Technological University \\
\texttt{Student ID: [to be filled]}"""

PREAMBLE = r"""\documentclass{article}

\usepackage[final]{neurips_2025}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{multirow}
\usepackage{tabularx}


\title{%TITLE%}

\author{%
%AUTHOR%
}

\begin{document}

\maketitle

"""

SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


UNICODE = [
    (r"ε(?:\\_|_)(floor|rank|stat|shift)", lambda m: r"$\varepsilon_{\mathrm{" + m.group(1) + "}}$"),
    ("ε", r"$\varepsilon$"),
    ("α", r"$\alpha$"),
    ("β", r"$\beta$"),
    ("γ", r"$\gamma$"),
    ("δ", r"$\delta$"),
    ("Δ", r"$\Delta$"),
    ("η", r"$\eta$"),
    ("λ", r"$\lambda$"),
    ("μ", r"$\mu$"),
    ("σ", r"$\sigma$"),
    ("τ", r"$\tau$"),
    ("θ", r"$\theta$"),
    ("×", r"$\times$"),
    ("≥", r"$\geq$"),
    ("≤", r"$\leq$"),
    ("→", r"$\rightarrow$"),
    ("±", r"$\pm$"),
    ("≈", r"$\approx$"),
    ("∼", r"$\sim$"),
    ("−", "-"),
    ("—", "---"),
    ("–", "--"),
    ("·", r"\textperiodcentered{}"),
    ("’", "'"),
    ("“", "``"),
    ("”", "''"),
]


def unicode_fix(text: str) -> str:
    for pattern, replacement in UNICODE:
        if callable(replacement):
            text = re.sub(pattern, replacement, text)
        else:
            text = text.replace(pattern, replacement)
    return text


def escape(text: str) -> str:
    """转义 LaTeX 特殊字符，但保留已经写好的数学片段。"""
    placeholders: dict[str, str] = {}

    def stash(pattern: str, replacement_prefix: str):
        def _sub(match: re.Match) -> str:
            key = f"{replacement_prefix}{len(placeholders)}"
            placeholders[key] = match.group(0)
            return key

        return _sub

    text = re.sub(r"\$[^$]+\$", stash(r"\$[^$]+\$", "@@MATH"), text)
    out = []
    for char in text:
        out.append(SPECIAL.get(char, char))
    text = "".join(out)
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def inline(text: str) -> str:
    """转义后处理粗体、斜体、行内代码，避免命令标记干扰转义判断。"""
    text = escape(text)
    text = unicode_fix(text)
    text = re.sub(r"`([^`]+)`", lambda m: r"\texttt{" + m.group(1) + "}", text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: r"\textbf{" + m.group(1) + "}", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: r"\emph{" + m.group(1) + "}", text)
    return text


def parse_table(lines: list[str]) -> tuple[str, list[str], list[str]]:
    header = [cell.strip() for cell in lines[0].strip().strip("|").split("|")]
    align_raw = [cell.strip() for cell in lines[1].strip().strip("|").split("|")]
    aligns = []
    for spec in align_raw:
        left, right = spec.startswith(":"), spec.endswith(":")
        aligns.append("c" if left and right else "r" if right else "l")
    rows = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        rows.append(cells[: len(header)])
    return header, aligns, rows


def _is_numeric_column(cells: list[list[str]], c: int) -> bool:
    values = [re.sub(r"[*`$\s]", "", row[c] if c < len(row) else "") for row in cells[1:]]
    values = [v for v in values if v]
    if not values:
        return False
    numeric = sum(1 for v in values if re.fullmatch(r"[-+\u2212]?[\d.,]+%?|\u2014|\u2013|[<>]?\d+(?:\.\d+)?", v))
    return numeric / len(values) >= 0.6


def _tabularx_spec(cells: list[list[str]], maxlen: list[int]) -> str:
    spec = []
    for c, length in enumerate(maxlen):
        if _is_numeric_column(cells, c):
            spec.append("r")
        elif length <= 12:
            spec.append("l")
        else:
            spec.append(">{\\raggedright\\arraybackslash}X")
    return "".join(spec)


def convert(markdown: str) -> str:
    lines = markdown.split("\n")
    body: list[str] = []
    index = 0
    abstract_done = False
    table_count = 0
    pending_caption = ""

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("# ") and not abstract_done:
            index += 1
            continue

        if stripped.startswith("## "):
            heading = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", stripped[3:].strip())
            index += 1
            if heading.lower() == "abstract":
                body.append("\\begin{abstract}")
                keywords = ""
                while index < len(lines) and not lines[index].strip().startswith("## "):
                    current = lines[index].strip()
                    if current.startswith("**Keywords"):
                        keywords = current
                    elif current and not current.startswith("---"):
                        body.append(inline(current) + "\n")
                    index += 1
                body.append("\\end{abstract}\n")
                if keywords:
                    body.append(inline(keywords) + "\n")
                abstract_done = True
                continue
            if heading.lower().startswith("references"):
                body.append("\\section*{References}")
                body.append("\\small")
                continue
            if heading.lower().startswith("appendix"):
                body.append("\\setcounter{table}{0}\\setcounter{figure}{0}")
                body.append("\\renewcommand{\\thetable}{A\\arabic{table}}")
                body.append("\\renewcommand{\\thefigure}{A\\arabic{figure}}")
                body.append(f"\\section*{{{inline(heading)}}}")
                continue
            body.append(f"\\section{{{inline(heading)}}}")
            continue

        if stripped.startswith("### "):
            subheading = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", stripped[4:].strip())
            body.append(f"\\subsection{{{inline(subheading)}}}")
            index += 1
            continue

        if re.match(r"^[-*]\s+", stripped):
            items = []
            while index < len(lines) and re.match(r"^[-*]\s+", lines[index].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[index].strip()))
                index += 1
            body.append("\\begin{itemize}")
            body.append("\\setlength{\\itemsep}{2pt}")
            for item in items:
                body.append(f"  \\item {inline(item)}")
            body.append("\\end{itemize}\n")
            continue

        if stripped.startswith("!["):
            match = re.match(r"!\[[^\]]*\]\(([^)]+)\)", stripped)
            index += 1
            caption = ""
            peek = index
            while peek < len(lines) and not lines[peek].strip():
                peek += 1
            if peek < len(lines) and lines[peek].strip().startswith("**Figure"):
                raw = lines[peek].strip()
                caption = re.sub(r"^\*\*Figure\s*[0-9A-Za-z]+\.\*\*\s*", "", raw)
                index = peek + 1
            if match:
                path = match.group(1)
                body.append("\\begin{figure}[htbp]")
                body.append("\\centering")
                body.append(f"\\includegraphics[width=0.92\\linewidth]{{{path}}}")
                body.append(f"\\caption{{{inline(caption)}}}")
                body.append("\\end{figure}\n")
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[index + 1].strip()):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index])
                index += 1
            header, aligns, rows = parse_table(block)
            table_count += 1
            caption, pending_caption = pending_caption, ""
            if caption:
                body.append("\\begin{table}[htbp]")
                body.append("\\centering")
                body.append(f"\\caption{{{inline(caption)}}}")
            else:
                body.append("\\begin{center}")
            body.append("\\small")
            cells = [header] + rows
            n_cols = len(aligns)
            maxlen = [
                max(len(re.sub(r"[*`$]", "", (row[c] if c < len(row) else ""))) for row in cells)
                for c in range(n_cols)
            ]

            mode = "plain"
            if n_cols >= 6:
                mode = "resize"
                body.append("\\resizebox{\\linewidth}{!}{%")
                body.append("\\begin{tabular}{" + "".join(aligns) + "}")
            else:
                spec = _tabularx_spec(cells, maxlen)
                if "X" in spec:
                    mode = "x"
                    body.append("\\begin{tabularx}{\\linewidth}{@{}" + spec + "@{}}")
                else:
                    body.append("\\begin{tabular}{" + "".join(aligns) + "}")

            body.append("\\toprule")
            body.append(" & ".join(inline(cell) for cell in header) + r" \\")
            body.append("\\midrule")
            for row in rows:
                body.append(" & ".join(inline(cell) for cell in row) + r" \\")
            body.append("\\bottomrule")

            if mode == "resize":
                body.append("\\end{tabular}}")
            elif mode == "x":
                body.append("\\end{tabularx}")
            else:
                body.append("\\end{tabular}")
            body.append("\\end{table}\n" if caption else "\\end{center}\n")
            continue

        if stripped.startswith("**Table"):
            inner = stripped.replace("**", "").strip()
            pending_caption = re.sub(r"^Table\s*[0-9A-Z]+\.[\s]*", "", inner).strip()
            index += 1
            continue

        if stripped.startswith("$$"):
            formula = stripped.strip("$")
            if formula:
                body.append("\\begin{equation}")
                body.append(formula)
                body.append("\\end{equation}")
            index += 1
            continue

        if stripped.startswith("---"):
            index += 1
            continue

        if re.match(r"^\d+\.\s", stripped):
            items = []
            while index < len(lines) and re.match(r"^\d+\.\s", lines[index].strip()):
                items.append(lines[index].strip()[3:])
                index += 1
            body.append("\\begin{enumerate}")
            body.append("\\setlength{\\itemsep}{1pt}")
            for item in items:
                body.append("  \\item " + inline(item))
            body.append("\\end{enumerate}\n")
            continue

        if stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
            body.append("\\noindent\\emph{" + inline(stripped.strip("*")) + "}\n")
            index += 1
            continue

        if not stripped or stripped.startswith("**Author:**"):
            index += 1
            continue

        body.append(inline(stripped) + "\n")
        index += 1

    return "\n".join(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="manuscript/MD6329_final_report.md")
    parser.add_argument("--output", default="manuscript/neurips/paper.tex")
    args = parser.parse_args()

    markdown = Path(args.input).read_text(encoding="utf-8")
    body = convert(markdown)
    head = PREAMBLE.replace("%TITLE%", TITLE).replace("%AUTHOR%", AUTHOR)
    tail = "\n\\end{document}\n"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + body + tail, encoding="utf-8")
    print(f"wrote {out} ({len(body.split())} words)")


if __name__ == "__main__":
    main()
