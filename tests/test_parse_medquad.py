"""Tests for the MedQuAD XML parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

from src.data.parse_medquad import parse_xml


def _write_xml(tmp_path: Path, content: str, name: str = "doc.xml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


DISORDERS_XML = """\
    <?xml version="1.0" encoding="UTF-8"?>
    <Document id="0000005_1" source="CancerGov" url="https://example.org/anal-cancer">
    <Focus>Anal Cancer</Focus>
    <FocusAnnotations>
        <UMLS>
            <CUIs>
                <CUI>C0279637</CUI>
                <CUI>C0153446</CUI>
            </CUIs>
            <SemanticTypes>
                <SemanticType>T191</SemanticType>
            </SemanticTypes>
            <SemanticGroup>Disorders</SemanticGroup>
        </UMLS>
    </FocusAnnotations>
    <QAPairs>
        <QAPair pid="1">
            <Question qid="0000005_1-1" qtype="information">What is Anal Cancer?</Question>
            <Answer>It is a disease.</Answer>
        </QAPair>
        <QAPair pid="2">
            <Question qid="0000005_1-2" qtype="symptoms">What are the symptoms?</Question>
            <Answer>Bleeding from the anus.</Answer>
        </QAPair>
    </QAPairs>
    </Document>
"""

DRUG_XML = """\
    <?xml version="1.0" encoding="UTF-8"?>
    <Document id="0001014" source="MPlusDrugs" url="https://example.org/prednisolone">
    <Focus>Prednisolone</Focus>
    <FocusAnnotations>
        <Category>Drug</Category>
    </FocusAnnotations>
    <QAPairs>
        <QAPair pid="1">
            <Question qid="0001014-1" qtype="indication">Who should get Prednisolone?</Question>
            <Answer></Answer>
        </QAPair>
        <QAPair pid="2">
            <Question qid="0001014-2" qtype="usage">How to use?</Question>
            <Answer>   </Answer>
        </QAPair>
    </QAPairs>
    </Document>
"""

MIXED_XML = """\
    <?xml version="1.0" encoding="UTF-8"?>
    <Document id="0000123" source="GARD" url="https://example.org/rare">
    <Focus>Rare disease</Focus>
    <FocusAnnotations>
        <UMLS>
            <CUIs><CUI>C0001</CUI></CUIs>
            <SemanticTypes><SemanticType>T047</SemanticType></SemanticTypes>
            <SemanticGroup>Disorders</SemanticGroup>
        </UMLS>
    </FocusAnnotations>
    <QAPairs>
        <QAPair pid="1">
            <Question qid="0000123-1" qtype="information">What is it?</Question>
            <Answer>It is rare.</Answer>
        </QAPair>
        <QAPair pid="2">
            <Question qid="0000123-2" qtype="causes">What causes it?</Question>
            <Answer></Answer>
        </QAPair>
        <QAPair pid="3">
            <Question qid="0000123-3" qtype="symptoms">Symptoms?</Question>
            <Answer>Fever.</Answer>
        </QAPair>
    </QAPairs>
    </Document>
"""

WHITESPACE_XML = """\
    <?xml version="1.0" encoding="UTF-8"?>
    <Document id="0000007" source="NHLBI" url="https://example.org/x">
    <Focus>X</Focus>
    <FocusAnnotations>
        <UMLS>
            <CUIs><CUI>C0X</CUI></CUIs>
            <SemanticTypes><SemanticType>T1</SemanticType></SemanticTypes>
            <SemanticGroup>Disorders</SemanticGroup>
        </UMLS>
    </FocusAnnotations>
    <QAPairs>
        <QAPair pid="1">
            <Question qid="0000007-1" qtype="information">  What
            is   X?  </Question>
            <Answer>
                Line one.

                Line two.
            </Answer>
        </QAPair>
    </QAPairs>
    </Document>
"""


def test_disorders_xml_two_records(tmp_path):
    path = _write_xml(tmp_path, DISORDERS_XML)
    records = parse_xml(path)

    assert len(records) == 2

    r1 = records[0]
    assert r1.qid == "0000005_1-1"
    assert r1.doc_id == "0000005_1"
    assert r1.source == "CancerGov"
    assert r1.url == "https://example.org/anal-cancer"
    assert r1.focus == "Anal Cancer"
    assert r1.cuis == ["C0279637", "C0153446"]
    assert r1.semantic_types == ["T191"]
    assert r1.semantic_group == "Disorders"
    assert r1.category is None
    assert r1.qtype == "information"
    assert r1.question == "What is Anal Cancer?"
    assert r1.answer == "It is a disease."

    r2 = records[1]
    assert r2.qid == "0000005_1-2"
    assert r2.qtype == "symptoms"


def test_drug_xml_all_filtered_out(tmp_path):
    """MPlusDrugs entries have empty <Answer> tags (copyright) → all skipped."""
    path = _write_xml(tmp_path, DRUG_XML)
    records = parse_xml(path)
    assert records == []


def test_mixed_xml_drops_only_empty_answer(tmp_path):
    path = _write_xml(tmp_path, MIXED_XML)
    records = parse_xml(path)

    assert len(records) == 2
    assert [r.qid for r in records] == ["0000123-1", "0000123-3"]
    assert all(r.semantic_group == "Disorders" for r in records)


def test_whitespace_normalised(tmp_path):
    path = _write_xml(tmp_path, WHITESPACE_XML)
    records = parse_xml(path)

    assert len(records) == 1
    assert records[0].question == "What is X?"
    assert records[0].answer == "Line one. Line two."
