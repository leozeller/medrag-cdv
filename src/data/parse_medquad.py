"""Parse MedQuAD XML corpus into a JSONL of QARecord entries.

MedQuAD ships as ~11k XML files in 12 subcorpora under data/medquad/.
Each XML is one source Document with one or more QAPairs. This script
walks the corpus, normalises whitespace, drops QAPairs with empty
<Answer> (e.g. the MPlusDrugs subset where answers are stripped for
copyright reasons), and writes one JSON record per QA pair to
data/processed/medquad.jsonl.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

from src.data.models import QARecord

_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str | None) -> str:
    if text is None:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _extract_cui_list(parent: ET.Element | None, child_tag: str) -> list[str]:
    if parent is None:
        return []
    return [
        el.text.strip()
        for el in parent.iter(child_tag)
        if el.text and el.text.strip()
    ]


def parse_xml(path: Path) -> list[QARecord]:
    """Parse a single MedQuAD XML file into zero or more QARecords.

    QAPairs with an empty <Answer> are skipped.
    """
    tree = ET.parse(path)
    doc = tree.getroot()

    doc_id = doc.get("id", "")
    source = doc.get("source", "")
    url = doc.get("url", "")
    focus = _normalise(doc.findtext("Focus"))

    annotations = doc.find("FocusAnnotations")
    umls = annotations.find("UMLS") if annotations is not None else None
    cuis = _extract_cui_list(umls, "CUI") if umls is not None else []
    semantic_types = _extract_cui_list(umls, "SemanticType") if umls is not None else []
    semantic_group = _normalise(umls.findtext("SemanticGroup")) if umls is not None else ""
    category = _normalise(annotations.findtext("Category")) if annotations is not None else ""

    records: list[QARecord] = []
    qa_pairs = doc.find("QAPairs")
    if qa_pairs is None:
        return records

    for pair in qa_pairs.findall("QAPair"):
        question_el = pair.find("Question")
        answer_el = pair.find("Answer")
        if question_el is None or answer_el is None:
            continue

        question = _normalise(question_el.text)
        answer = _normalise(answer_el.text)
        if not answer:
            continue

        records.append(
            QARecord(
                qid=question_el.get("qid", ""),
                doc_id=doc_id,
                source=source,
                url=url,
                focus=focus,
                cuis=cuis,
                semantic_types=semantic_types,
                semantic_group=semantic_group or None,
                category=category or None,
                qtype=question_el.get("qtype", ""),
                question=question,
                answer=answer,
            )
        )

    return records


def parse_corpus(root: Path) -> Iterator[QARecord]:
    """Walk all XML files under `root` recursively and yield QARecords."""
    for path in sorted(root.rglob("*.xml")):
        try:
            yield from parse_xml(path)
        except ET.ParseError as exc:
            print(f"WARN: failed to parse {path}: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/medquad"),
        help="Root of MedQuAD XML corpus",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/medquad.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for record in parse_corpus(args.input):
            fh.write(record.model_dump_json())
            fh.write("\n")
            count += 1

    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
