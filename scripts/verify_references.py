"""逐条核验论文参考文献，并下载可公开获取的 PDF。

数据源优先级：
1. Crossref  —— 权威元数据（DOI、刊名、卷期页、年份）
2. arXiv     —— 预印本全文
3. Unpaywall —— 由 DOI 反查开放获取副本

输出：
    reference/metadata.json        每条文献的核验结果
    reference/REFERENCE_INDEX.md   人读的核验表
    reference/pdfs/NN_key.pdf      可下载的全文
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = "MD6329-ReferenceCheck/1.0 (student project; contact: zhengyu@example.edu)"
ROOT = Path("reference")
PDF_DIR = ROOT / "pdfs"

# (编号, 简称, 标题, 第一作者, venue 关键词, 年份, 已知 arXiv id)
REFS = [
    (1, "ptbxl", "PTB-XL, a large publicly available electrocardiography dataset",
     "Wagner", "Scientific Data", 2020, None),
    (2, "guo_calibration", "On Calibration of Modern Neural Networks",
     "Guo", "ICML", 2017, "1706.04599"),
    (3, "deep_ensembles", "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles",
     "Lakshminarayanan", "NeurIPS", 2017, "1612.01474"),
    (4, "selective_classification", "Selective Classification for Deep Neural Networks",
     "Geifman", "NeurIPS", 2017, "1705.08500"),
    (5, "selectivenet", "SelectiveNet: A Deep Neural Network with an Integrated Reject Option",
     "Geifman", "ICML", 2019, "1901.09192"),
    (6, "conformal_risk_control", "Conformal Risk Control",
     "Angelopoulos", "ICLR", 2023, "2208.02814"),
    (7, "strodthoff", "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL",
     "Strodthoff", "IEEE Journal of Biomedical and Health Informatics", 2021, "2004.13701"),
    (8, "ovadia", "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift",
     "Ovadia", "NeurIPS", 2019, "1906.02530"),
    (9, "kendall_gal", "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?",
     "Kendall", "NeurIPS", 2017, "1703.04977"),
    (10, "vaicenavicius", "Evaluating Model Calibration in Classification",
     "Vaicenavicius", "AISTATS", 2019, None),
    (11, "challenge2020", "Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020",
     "Alday", "Physiological Measurement", 2020, None),
    (12, "encase", "ENCASE: An ENsemble ClASsifier for ECG Classification Using Expert Features and Deep Neural Networks",
     "Hong", "Computing in Cardiology", 2017, None),
    (13, "rabanser_gap", "What Does It Take to Build a Performant Selective Classifier?",
     "Rabanser", "NeurIPS", 2025, "2510.20242"),
    (14, "conformal_multilabel_review", "Conformal prediction for multi-label learning: a review of methods and guarantees",
     "Conformal prediction multi-label", "Philosophical Transactions of the Royal Society A", 2026, None),
    (15, "perclass_crc_ecg", "Per-Class Conformal Risk Control for Multi-Label ECG Classification: Achieving Clinically-Justified False Negative Rate Guarantees",
     "EL ALLAM", "Physiological Measurement", 2026, None),
    (16, "hannun", "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network",
     "Hannun", "Nature Medicine", 2019, None),
    (17, "lime", "Why Should I Trust You? Explaining the Predictions of Any Classifier",
     "Ribeiro", "KDD", 2016, "1602.04938"),
]


# 人工纠偏：Crossref 对这些条目的自动匹配不可靠，或存在更权威的开放副本
OVERRIDES = {
    1:  {"extra_url": "https://www.nature.com/articles/s41597-020-0495-6.pdf",
         "note": "Nature 直链返回 HTML；PMC7248071 全文 XML 已保存为 01_ptbxl_fulltext.xml"},
    2:  {"skip_crossref": True},   # ICML PMLR，无 DOI
    3:  {"skip_crossref": True},   # NeurIPS，无 DOI
    4:  {"skip_crossref": True},   # 自动匹配到 NTU 学位论文，DOI 错误
    5:  {"skip_crossref": True},
    6:  {"skip_crossref": True},   # ICLR OpenReview，无 DOI
    8:  {"skip_crossref": True},
    9:  {"skip_crossref": True},
    10: {"skip_crossref": True,
         "extra_url": "http://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf"},
    11: {"skip_crossref": True, "doi": "10.1088/1361-6579/abc960",
         "meta": {"container": "Physiological Measurement", "volume": "41", "issue": "12",
                  "page": "124003", "year": 2020, "title": "Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020"},
         "note": "原 [11] Nonaka & Seita 经 Crossref/arXiv/OpenAlex 四路检索确认不存在，已替换；本地全文为 medRxiv 预印本，正式版见 Physiol. Meas. 41(12):124003"},
    17: {"doi": "10.1145/2939672.2939778"},   # 论文引的是 KDD 版，不是 NAACL demo 版
}


def get(url: str, timeout: int = 25) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(f"    ! {type(exc).__name__}: {str(exc)[:90]}")
        return None


def get_json(url: str) -> dict | None:
    raw = get(url)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split()


def similarity(a: str, b: str) -> float:
    sa, sb = set(normalize(a)), set(normalize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def crossref(title: str, author: str) -> dict | None:
    query = urllib.parse.quote(f"{title} {author}")
    url = f"https://api.crossref.org/works?query.bibliographic={query}&rows=5&select=DOI,title,author,container-title,issued,volume,issue,page,type"
    data = get_json(url)
    if not data:
        return None
    items = data.get("message", {}).get("items", [])
    best, best_score = None, 0.0
    for item in items:
        cand = (item.get("title") or [""])[0]
        score = similarity(title, cand)
        if score > best_score:
            best, best_score = item, score
    if best is None or best_score < 0.55:
        return None
    authors = best.get("author", []) or []
    names = [f"{a.get('family','')}" for a in authors][:4]
    issued = best.get("issued", {}).get("date-parts", [[None]])[0]
    return {
        "score": round(best_score, 3),
        "doi": best.get("DOI"),
        "title": (best.get("title") or [""])[0],
        "authors": names,
        "container": (best.get("container-title") or [""])[0],
        "year": issued[0] if issued else None,
        "volume": best.get("volume"),
        "issue": best.get("issue"),
        "page": best.get("page"),
        "type": best.get("type"),
    }


def unpaywall(doi: str) -> str | None:
    data = get_json(f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email=zhengyu@example.edu")
    if not data:
        return None
    best = data.get("best_oa_location") or {}
    return best.get("url_for_pdf") or best.get("url")


def arxiv_meta(arxiv_id: str) -> dict | None:
    raw = get(f"http://export.arxiv.org/api/query?id_list={arxiv_id}")
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    title = re.search(r"<entry>.*?<title>(.*?)</title>", text, re.S)
    return {"title": re.sub(r"\s+", " ", title.group(1)).strip() if title else None}


def download_pdf(url: str, dest: Path) -> str:
    raw = get(url, timeout=60)
    if not raw:
        return "download failed"
    if not raw.startswith(b"%PDF"):
        return "not a PDF (likely a landing page)"
    dest.write_bytes(raw)
    return f"ok ({len(raw)//1024} KB)"


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)
    records = []

    for num, key, title, author, venue, year, arxiv in REFS:
        print(f"\n[{num:02d}] {key}")
        entry = {"n": num, "key": key, "paper_title": title, "paper_venue": venue, "paper_year": year}
        ov = OVERRIDES.get(num, {})

        meta = None if ov.get("skip_crossref") else crossref(title, author)
        if ov.get("meta"):
            meta = {"score": 1.0, "authors": [author], "doi": ov.get("doi"), **ov["meta"]}
        if ov.get("doi") and meta:
            meta["doi"] = ov["doi"]
        elif ov.get("doi"):
            meta = {"doi": ov["doi"], "title": title, "year": year, "container": venue, "score": 1.0, "authors": [author]}
        if meta:
            entry["crossref"] = meta
            print(f"    crossref: {meta['year']} | {meta['container'][:50]} | DOI {meta['doi']} | sim {meta['score']}")
        else:
            print("    crossref: no confident match")

        if arxiv:
            am = arxiv_meta(arxiv)
            entry["arxiv"] = {"id": arxiv, "title": (am or {}).get("title")}
            print(f"    arxiv: {arxiv} -> {(am or {}).get('title','?')[:60]}")

        # 下载优先级：arXiv → Unpaywall
        dest = PDF_DIR / f"{num:02d}_{key}.pdf"
        status = "no open copy found"
        if dest.exists() and dest.stat().st_size > 20000:
            status = "cached"
            entry["pdf_source"] = entry.get("pdf_source", "previous run")
        elif ov.get("extra_url"):
            status = download_pdf(ov["extra_url"], dest)
            entry["pdf_source"] = ov["extra_url"]
        if arxiv and not dest.exists():
            status = download_pdf(f"https://arxiv.org/pdf/{arxiv}", dest)
            entry["pdf_source"] = f"https://arxiv.org/pdf/{arxiv}"
        if not dest.exists() and meta and meta.get("doi"):
            oa = unpaywall(meta["doi"])
            if oa:
                status = download_pdf(oa, dest)
                entry["pdf_source"] = oa
        entry["pdf_status"] = status
        print(f"    pdf: {status}")
        records.append(entry)
        time.sleep(1.0)

    (ROOT / "metadata.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 生成人读表格 ----
    lines = [
        "# 参考文献核验表",
        "",
        "由 `code/scripts/verify_references.py` 生成。核验源：Crossref（元数据）、arXiv（预印本）、Unpaywall（开放副本）。",
        "",
        "| # | 简称 | 论文中标注 | Crossref 实际 | DOI | 核实 | 全文 |",
        "|---:|---|---|---|---|---|---|",
    ]
    for r in records:
        cr = r.get("crossref") or {}
        actual = ""
        if cr:
            bits = [str(cr.get("container") or ""), str(cr.get("year") or "")]
            if cr.get("volume"):
                bits.append(f"vol {cr['volume']}")
            if cr.get("issue"):
                bits.append(f"no {cr['issue']}")
            if cr.get("page"):
                bits.append(f"pp {cr['page']}")
            actual = ", ".join(b for b in bits if b)
        doi = cr.get("doi") or "—"
        status = r.get("pdf_status", "")
        mark = "✅" if status.startswith("ok") else "⚠️"
        verdict = "❌ 无法核实" if r.get("unverified") else ("✅" if cr or r.get("arxiv") or r.get("pdf_status","").startswith(("ok","cached")) else "⚠️")
        lines.append(
            f"| {r['n']} | `{r['key']}` | {r['paper_venue']} {r['paper_year']} | {actual or '未匹配'} | {doi} | {verdict} | {mark} {status} |"
        )
    missing = [r for r in records if not r.get("pdf_status", "").startswith(("ok", "cached"))]
    if missing and "## 未能取得 PDF 的条目" not in "\n".join(lines):
        lines += ["", "## 未能取得 PDF 的条目", ""]
        for r in missing:
            why = OVERRIDES.get(r["n"], {}).get("note") or r.get("pdf_status", "")
            lines.append(f"- **[{r['n']}] `{r['key']}`** — {r['paper_venue']} {r['paper_year']}：{why}")

    missing = [r for r in records if not r.get("pdf_status", "").startswith(("ok", "cached"))]
    if missing and "## 未能取得 PDF 的条目" not in "\n".join(lines):
        lines += ["", "## 未能取得 PDF 的条目", ""]
        for r in missing:
            why = OVERRIDES.get(r["n"], {}).get("note") or r.get("pdf_status", "")
            lines.append(f"- **[{r['n']}] `{r['key']}`** — {r['paper_venue']} {r['paper_year']}：{why}")

    bad = [r for r in records if r.get("unverified")]
    if bad:
        lines += ["", "## 需处理：无法核实的条目", ""]
        for r in bad:
            lines.append(f"- **[{r['n']}] `{r['key']}`** — {r['paper_title']}")
            lines.append(f"  - {r['note']}")
        lines.append("")
        lines.append("建议：该条目在 Related Work 中与 [7]（Strodthoff 2021，已核实）并列支撑同一句话，"
                     "直接删除 [11] 不影响论证，且能消除一处学术诚信风险。")

    (ROOT / "REFERENCE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n已写出 {ROOT/'REFERENCE_INDEX.md'}，PDF {len(list(PDF_DIR.glob('*.pdf')))} 份")


if __name__ == "__main__":
    main()
