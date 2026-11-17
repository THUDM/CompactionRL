import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from harbor.models.trial.result import ProfileSpan, TrialResult


_CATEGORY_COLORS = {
    "startup": "#4e79a7",
    "generate": "#59a14f",
    "summary_generate": "#b07aa1",
    "tool_call": "#f28e2b",
    "response_processing": "#76b7b2",
    "observation_processing": "#edc948",
    "context_management": "#af7aa1",
    "agent_bootstrap": "#9c755f",
    "agent_finalize": "#bab0ab",
    "verify": "#e15759",
    "artifacts": "#ff9da7",
    "cleanup": "#79706e",
}


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _duration_sec(started_at: datetime | None, finished_at: datetime | None) -> float:
    if started_at is None or finished_at is None:
        return 0.0
    return max((finished_at - started_at).total_seconds(), 0.0)


def _trial_status(trial_result: TrialResult) -> str:
    if trial_result.exception_info is not None:
        return "failed"
    if trial_result.verifier_result is not None:
        return "verified"
    return "completed"


def build_trial_profile_record(trial_result: TrialResult) -> dict[str, Any]:
    trial_started_at = trial_result.started_at
    spans = sorted(
        trial_result.profiling_spans or [],
        key=lambda span: (span.started_at, span.finished_at, span.label),
    )
    normalized_spans: list[ProfileSpan] = []
    for span in spans:
        category = span.category
        if category == "generate" and span.label.startswith("summary:"):
            category = "summary_generate"
        normalized_spans.append(
            ProfileSpan(
                category=category,
                label=span.label,
                started_at=span.started_at,
                finished_at=span.finished_at,
                metadata=span.metadata,
            )
        )

    completed_spans = _fill_gaps_with_inferred_spans(
        normalized_spans,
        trial_started_at=trial_result.started_at,
        trial_finished_at=trial_result.finished_at,
    )

    span_records: list[dict[str, Any]] = []
    for span in completed_spans:
        if trial_started_at is None:
            offset_start_sec = 0.0
        else:
            offset_start_sec = max(
                (span.started_at - trial_started_at).total_seconds(),
                0.0,
            )
        span_records.append(
            {
                "category": span.category,
                "label": span.label,
                "started_at": _iso(span.started_at),
                "finished_at": _iso(span.finished_at),
                "offset_start_sec": offset_start_sec,
                "duration_sec": _duration_sec(span.started_at, span.finished_at),
                "metadata": span.metadata,
            }
        )

    return {
        "trial_name": trial_result.trial_name,
        "task_name": trial_result.task_name,
        "status": _trial_status(trial_result),
        "started_at": _iso(trial_result.started_at),
        "finished_at": _iso(trial_result.finished_at),
        "duration_sec": _duration_sec(trial_result.started_at, trial_result.finished_at),
        "spans": span_records,
    }


def _inferred_gap_category(
    prev_span: ProfileSpan | None,
    next_span: ProfileSpan | None,
) -> tuple[str, str]:
    prev_category = prev_span.category if prev_span is not None else None
    next_category = next_span.category if next_span is not None else None

    if prev_span is None and next_category == "startup":
        return "startup", "startup_prepare"
    if prev_span is None and next_category in {"generate", "summary_generate", "tool_call"}:
        return "agent_bootstrap", "agent_bootstrap"
    if prev_category == "startup" and next_category == "startup":
        return "startup", "startup_transition"
    if prev_category == "startup" and next_category in {"generate", "summary_generate", "tool_call"}:
        return "agent_bootstrap", "agent_bootstrap"
    if prev_category in {"generate", "summary_generate"} and next_category == "tool_call":
        return "response_processing", "response_processing"
    if prev_category == "tool_call" and next_category in {"generate", "summary_generate"}:
        return "observation_processing", "observation_processing"
    if prev_category in {"summary_generate", "context_management"} and next_category in {"generate", "summary_generate"}:
        return "context_management", "context_management"
    if prev_category == "tool_call" and next_category == "verify":
        return "agent_finalize", "agent_finalize"
    if prev_category in {"generate", "summary_generate"} and next_category == "verify":
        return "agent_finalize", "agent_finalize"
    if prev_category == "verify":
        return "cleanup", "cleanup_wait"
    if next_category == "verify":
        return "agent_finalize", "agent_finalize"
    return "response_processing", "response_processing"


def _fill_gaps_with_inferred_spans(
    spans: list[ProfileSpan],
    trial_started_at: datetime | None,
    trial_finished_at: datetime | None,
) -> list[ProfileSpan]:
    if trial_started_at is None or trial_finished_at is None:
        return spans

    spans = sorted(spans, key=lambda span: (span.started_at, span.finished_at, span.label))
    completed: list[ProfileSpan] = []
    cursor = trial_started_at
    previous: ProfileSpan | None = None

    for span in spans:
        if span.started_at > cursor:
            category, label = _inferred_gap_category(previous, span)
            completed.append(
                ProfileSpan(
                    category=category,
                    label=label,
                    started_at=cursor,
                    finished_at=span.started_at,
                )
            )
        completed.append(span)
        if span.finished_at > cursor:
            cursor = span.finished_at
        previous = span

    if cursor < trial_finished_at:
        category, label = _inferred_gap_category(previous, None)
        completed.append(
            ProfileSpan(
                category=category,
                label=label if previous is not None else "trial_finalize",
                started_at=cursor,
                finished_at=trial_finished_at,
            )
        )

    return completed


def _jsonl_dump(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + (
        "\n" if records else ""
    )


def _render_svg(records: list[dict[str, Any]]) -> str:
    label_width = 360
    chart_width = 1400
    row_height = 24
    top_padding = 40
    bottom_padding = 24
    max_duration = max((record["duration_sec"] for record in records), default=1.0)
    max_duration = max(max_duration, 1.0)
    scale = chart_width / max_duration

    total_height = top_padding + bottom_padding + row_height * max(len(records), 1)
    total_width = label_width + chart_width + 40
    svg_parts = [
        f'<svg width="{total_width}" height="{total_height}" viewBox="0 0 {total_width} {total_height}" xmlns="http://www.w3.org/2000/svg">'
    ]
    svg_parts.append(
        f'<rect x="0" y="0" width="{total_width}" height="{total_height}" fill="#ffffff" />'
    )
    svg_parts.append(
        f'<line x1="{label_width}" y1="18" x2="{label_width + chart_width}" y2="18" stroke="#888" stroke-width="1" />'
    )

    tick_count = 8
    for tick in range(tick_count + 1):
        tick_sec = max_duration * tick / tick_count
        x = label_width + tick_sec * scale
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="18" x2="{x:.2f}" y2="{total_height - bottom_padding}" stroke="#e5e7eb" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="14" font-size="11" text-anchor="middle" fill="#555">{tick_sec:.1f}s</text>'
        )

    for idx, record in enumerate(records):
        y = top_padding + idx * row_height
        label = f'{record["task_name"]} [{record["status"]}]'
        svg_parts.append(
            f'<text x="8" y="{y + 15}" font-size="11" fill="#222">{escape(label)}</text>'
        )
        svg_parts.append(
            f'<text x="{label_width - 8}" y="{y + 15}" font-size="10" text-anchor="end" fill="#666">{record["duration_sec"]:.1f}s</text>'
        )
        svg_parts.append(
            f'<line x1="{label_width}" y1="{y + row_height - 3}" x2="{label_width + chart_width}" y2="{y + row_height - 3}" stroke="#f3f4f6" stroke-width="1" />'
        )
        if not record["spans"]:
            continue
        for span in record["spans"]:
            width = max(span["duration_sec"] * scale, 1.5)
            x = label_width + span["offset_start_sec"] * scale
            color = _CATEGORY_COLORS.get(span["category"], "#9ca3af")
            title = (
                f'{record["trial_name"]}\n'
                f'{span["label"]}\n'
                f'{span["category"]}: {span["duration_sec"]:.3f}s'
            )
            svg_parts.append(
                f'<rect x="{x:.2f}" y="{y + 5}" width="{width:.2f}" height="12" rx="2" fill="{color}"><title>{escape(title)}</title></rect>'
            )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _render_html(records: list[dict[str, Any]]) -> str:
    legend_items = "".join(
        (
            '<span class="legend-item">'
            f'<span class="swatch" style="background:{color}"></span>{escape(category)}'
            "</span>"
        )
        for category, color in _CATEGORY_COLORS.items()
    )
    svg = _render_svg(records)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Trial Profiling Timeline</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; color: #111827; }}
    .legend {{ display: flex; gap: 16px; margin: 8px 0 16px; flex-wrap: wrap; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 2px; display: inline-block; }}
    .meta {{ color: #4b5563; font-size: 13px; margin-bottom: 12px; }}
    .chart {{ border: 1px solid #e5e7eb; overflow: auto; max-width: 100%; }}
  </style>
</head>
<body>
  <h1>Trial Profiling Timeline</h1>
  <div class="meta">One row per trial. Segments are positioned by elapsed time since that trial started.</div>
  <div class="legend">{legend_items}</div>
  <div class="chart">{svg}</div>
</body>
</html>
"""


def write_job_profile_artifacts(job_dir: Path, trial_results: list[TrialResult]) -> None:
    records = [build_trial_profile_record(trial_result) for trial_result in trial_results]
    records.sort(key=lambda record: (record["started_at"] or "", record["trial_name"]))

    (job_dir / "profiling_timeline.jsonl").write_text(_jsonl_dump(records))
    (job_dir / "profiling_timeline.html").write_text(_render_html(records))
