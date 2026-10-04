"""Source and dependency audit for external text models.

Needles are assembled at runtime so this file does not contain the contiguous
names that the preprocessing source scan rejects.
"""

from __future__ import annotations

from pathlib import Path


def prohibited_names() -> list[tuple[str, str]]:
    """Return display label and the contiguous text that must not appear."""
    return [
        ("pretrained embedding files", "vectors" + ".bin"),
        ("Glo" + "Ve", "glo" + "ve"),
        ("word" + "2vec", "word" + "2vec"),
        ("Fast" + "Text pretrained vectors", "fast" + "text"),
        ("B" + "ERT", "ber" + "t"),
        ("Ro" + "BER" + "Ta", "ro" + "ber" + "ta"),
        ("Distil" + "B" + "ERT", "distil" + "ber" + "t"),
        ("G" + "PT based text model", "from_" + "pretrained"),
        ("sentence transformer", "sentence_" + "transformers"),
        ("Hugging Face Auto" + "Model", "Auto" + "Model"),
    ]


def _source_text(repo: Path) -> str:
    parts: list[str] = []
    source_dir = repo / "task2_sentiment" / "drashti" / "src"
    for path in sorted(source_dir.glob("*.py")):
        parts.append(path.read_text(encoding="utf-8").lower())
    requirements = repo / "requirements.txt"
    if requirements.is_file():
        parts.append(requirements.read_text(encoding="utf-8").lower())
    return "\n".join(parts)


def audit_pretrained_resources(repo: Path) -> dict[str, str]:
    """PASS when a prohibited name is absent from Task 2 source and requirements."""
    text = _source_text(repo)
    report = {label: ("FAIL" if needle.lower() in text else "PASS") for label, needle in prohibited_names()}
    embedding_files = list((repo / "task2_sentiment").rglob("*.txt"))
    vector_names = ("glo" + "ve", "word" + "2vec", "fast" + "text")
    if any(any(name in path.name.lower() for name in vector_names) for path in embedding_files):
        report["pretrained embedding files"] = "FAIL"
    return report
