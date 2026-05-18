"""engine_v4.types — shared dataclasses for every layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

InputKind = Literal["video", "audio", "pdf", "text", "url", "auto"]


@dataclass
class SourceSpec:
    location: str
    kind: InputKind = "auto"
    title: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass
class TranscriptWindow:
    start_sec: float
    end_sec: float
    text: str
    is_high_info: bool = False
    keyword_hits: int = 0


@dataclass
class Frame:
    timestamp_sec: float
    path: Path
    phash: int = 0
    caption: str = ""


@dataclass
class Source:
    source_id: str
    spec: SourceSpec
    workspace: Path
    media_path: Optional[Path]
    duration_sec: float
    segments: list[TranscriptSegment] = field(default_factory=list)
    windows: list[TranscriptWindow] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutlineSection:
    id: str
    title: str
    bloom_level: str = "understand"
    key_terms: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    prereqs: list[str] = field(default_factory=list)


@dataclass
class Outline:
    sections: list[OutlineSection]
    target_count: int = 0


@dataclass
class Citation:
    ref_str: str
    title: str
    authors: str = ""
    year: str = ""
    url: str = ""


@dataclass
class Chapter:
    id: str
    title: str
    body_md: str
    citations: list[Citation] = field(default_factory=list)
    eval_score: float = 0.0
    approved: bool = False
    figures: list[Frame] = field(default_factory=list)


@dataclass
class PipelineState:
    spec: SourceSpec
    run_workspace: Path
    source: Optional[Source] = None
    outline: Optional[Outline] = None
    chapters: list[Chapter] = field(default_factory=list)


@dataclass
class BuildResult:
    run_id: str
    workspace: Path
    states: list[PipelineState]
    textbook_md: Optional[Path] = None
    epub: Optional[Path] = None
    html: Optional[Path] = None
    pdf: Optional[Path] = None
    errors: list[str] = field(default_factory=list)
    # chapter_groups: [{"chapter_title": str, "state_indices": [int, ...]}]
    chapter_groups: list[dict] = field(default_factory=list)
