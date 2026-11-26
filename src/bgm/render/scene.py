"""Scene brief for renders: what happens on screen -> what the music should say (user request, D101).

A composer reads the scene before choosing music. Here a frozen vision-language model (Qwen2.5-VL-3B-Instruct, fp16,
eval, no_grad) looks at a few frames of each segment of the SILENT 25 fps video and drafts {scene, emotion, music}
into an editable YAML file (renders/<name>.brief.yaml). The user can correct any line; an existing file is never
redrafted. At render time each segment's "<emotion>. <music>" text is encoded with the CLAP text tower and scored
against the music windows' CLAP audio embeddings (same 512-d space) as the branch "brief", weight
`render.scene_brief.weight`. Off by default; only `bgm render` / `bgm brief` read it.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import yaml

from bgm.render.branches import BranchIndex

log = logging.getLogger(__name__)
EMOTIONS = ("heroic", "action", "tension", "fear", "sad", "romantic", "happy", "comic", "peaceful", "wonder")
SUMMARY_PROMPT = (
    "You are the music director of an Indian film. These {n} frames are sampled in order across a whole {dur:.0f} s "
    "sequence. In English, in at most 40 words: what is the sequence about, and what overall feeling should its music "
    "carry? Plain text, no list."
)
PROMPT = (
    "You are the music director of an Indian film, scoring this sequence: {summary}\n"
    "These {n} frames are one moment of it, {a:.0f}-{b:.0f} s in. The music so far was: {prev}. A composer keeps the "
    "same emotion through a scene and changes it only when the story clearly turns (not for a single calm-looking shot).\n"
    "Reply in English with JSON only, keys:\n"
    '"scene": one sentence, what happens in THESE frames;\n'
    '"emotion": the feeling the audience should get, one of ' + ", ".join(EMOTIONS) + ";\n"
    '"music": background music for this moment in at most 20 words: mood, energy, tempo, instruments;\n'
    '"needs_music": "yes" or "no". Answer "no" only if a film composer would leave this moment WITHOUT background music '
    '(quiet dialogue where music would distract, a deliberate silent beat before a shock); most moments need music.'
)


def sample_times(a: float, b: float, n_max: int = 4) -> np.ndarray:
    """Evenly spaced frame times strictly inside [a, b] s: about one per 2 s, 1..n_max."""
    n = int(np.clip(round((b - a) / 2), 1, n_max))
    return np.linspace(a, b, n + 2)[1:-1]


def read_frames(silent_video: Path, times: np.ndarray, max_side: int = 448) -> list:
    """RGB PIL frames of the silent interim video at `times` s (rule 8: never the original soundtrack's file)."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(silent_video))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        s = max_side / max(h, w)
        out.append(Image.fromarray(cv2.cvtColor(cv2.resize(f, (round(w * s), round(h * s))), cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


def parse_reply(text: str) -> dict:
    """The model's JSON reply -> {scene, emotion, music}; tolerant of code fences and unknown emotions."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    emo = str(d.get("emotion", "")).strip().lower()
    nm = str(d.get("needs_music", "yes")).strip().lower()
    return {"scene": str(d.get("scene", "")).strip(), "emotion": emo if emo in EMOTIONS else (emo or "peaceful"),
            "music": str(d.get("music", "")).strip(), "needs_music": nm not in ("no", "false", "0", "none")}


class QwenVL:
    """Frozen Qwen2.5-VL describer: `describe(frames, a, b, prev) -> {scene, emotion, music}`."""

    def __init__(self, model_id: str, revision: str | None = None, device: str = "cuda", max_new_tokens: int = 120):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch, self.device, self.max_new = torch, device, max_new_tokens
        dt = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, revision=revision, dtype=dt).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id, revision=revision)

    def _ask(self, frames: list, prompt: str, max_new: int) -> str:
        content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        x = self.proc(text=[text], images=frames, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            y = self.model.generate(**x, max_new_tokens=max_new, do_sample=False)
        return self.proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def summarize(self, frames: list, dur: float) -> str:
        return self._ask(frames, SUMMARY_PROMPT.format(n=len(frames), dur=dur), 90)

    def describe(self, frames: list, a: float, b: float, summary: str = "", prev: str = "") -> dict:
        prompt = PROMPT.format(summary=summary or "(unknown)", n=len(frames), a=a, b=b, prev=prev or "nothing yet (this is the start)")
        return parse_reply(self._ask(frames, prompt, self.max_new))


class FakeDescriber:
    """Deterministic stand-in for tests: emotion from the span's position."""

    def summarize(self, frames: list, dur: float) -> str:
        return f"{len(frames)} frames over {dur:.0f} s"

    def describe(self, frames: list, a: float, b: float, summary: str = "", prev: str = "") -> dict:
        e = EMOTIONS[int(a) % len(EMOTIONS)]
        return {"scene": f"{len(frames)} frames at {a:.0f}-{b:.0f} s", "emotion": e, "music": f"{e} background music",
                "needs_music": int(a) % 3 != 1}


def draft_brief(describer, silent_video: Path, segments: list[tuple[float, float]], n_summary: int = 12) -> tuple[str, list[dict]]:
    """Whole-sequence summary first (n_summary frames across it), then each segment in its light.
    Returns (summary, [{start_s, end_s, scene, emotion, music}])."""
    dur = segments[-1][1] - segments[0][0]
    summary = describer.summarize(read_frames(silent_video, np.linspace(segments[0][0], segments[-1][1], n_summary + 2)[1:-1]), dur)
    log.info("brief summary: %s", summary)
    out, prev = [], ""
    for a, b in segments:
        d = describer.describe(read_frames(silent_video, sample_times(a, b)), a, b, summary, prev)
        out.append({"start_s": round(float(a), 2), "end_s": round(float(b), 2), **d})
        prev = f"{d['emotion']} ({d['music']})"
        log.info("brief %6.1f-%6.1f s  %-9s %s %s | %s", a, b, d["emotion"], "" if d["needs_music"] else "[NO MUSIC]", d["scene"], d["music"])
    return summary, out


def write_brief(path: Path, brief: list[dict], video: str, summary: str = "") -> None:
    head = (f"# Scene brief for {video}\n# Edit any scene / emotion / music line, then re-run the same render command;\n"
            f"# this file is reused as is. Emotions: {', '.join(EMOTIONS)}\n"
            f"# needs_music: false (or music: none) leaves that part without background music (D107)\n" + (f"# Summary: {summary}\n" if summary else ""))
    path.write_text(head + yaml.safe_dump(brief, sort_keys=False, allow_unicode=True, width=140))


def read_brief(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text())


def brief_text(e: dict) -> str:
    """CLAP text for one brief entry."""
    return f"{e['emotion']} film background music. {e['music']}".strip()


def span_weights(brief: list[dict], a: float, b: float) -> np.ndarray:
    """Weights [E] of the brief entries for video span [a, b] s: their overlap with it (the nearest entry if none)."""
    ov = np.array([max(0.0, min(b, e["end_s"]) - max(a, e["start_s"])) for e in brief])
    if ov.sum() <= 0:
        ov = np.zeros(len(brief))
        ov[int(np.argmin([abs((e["start_s"] + e["end_s"]) / 2 - (a + b) / 2) for e in brief]))] = 1.0
    return ov / ov.sum()


class BriefIndex(BranchIndex):
    """Adds the "brief" branch: cos(CLAP text of the segment's brief, the window's mean CLAP audio embedding)."""
    name = "brief"

    def __init__(self, index, brief: list[dict], text_emb: np.ndarray):
        super().__init__(index)
        self.brief, self.t = brief, text_emb                 # text_emb [E, 512] unit, one per brief entry
        clap = [it.sem[0] for it in index.items]             # MUSIC_KEYS[0] = sem_clap_1hz
        whole = np.stack([p.means(np.array([0.0]), np.array([it.duration]))[0] for p, it in zip(clap, index.items)])
        self.clap, self.g_clap = clap, whole / np.linalg.norm(whole, axis=1, keepdims=True)

    def _t(self, a: float, b: float) -> np.ndarray:
        t = span_weights(self.brief, a, b) @ self.t
        return t / np.linalg.norm(t)

    def window_branch(self, a, b, fi, st, d):
        out = np.empty(len(fi))
        for i in np.unique(fi):
            sel = fi == i
            dur = self.items[i].duration
            e = self.clap[i].means(np.clip(st[sel], 0, dur), np.clip(st[sel] + d, 0, dur))
            out[sel] = (e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)) @ self._t(a, b)
        return out

    def file_branch(self, a, b):
        return self.g_clap @ self._t(a, b)


def brief_path(cfg, out: Path) -> Path:
    f = (cfg.render.get("scene_brief") or {}).get("file")
    return Path(f) if f else Path(cfg.paths.renders) / f"{out.stem}.brief.yaml"


def ensure_brief(cfg, path: Path, silent_video: Path, segments, video: str) -> list[dict]:
    """The brief at `path`, drafted with the VLM first if the file does not exist yet."""
    if path.exists():
        log.info("scene brief: using %s (edit it to steer the music)", path)
        return read_brief(path)
    sb = cfg.render.scene_brief
    import torch
    vlm = QwenVL(sb.vlm_model, sb.get("vlm_revision"), "cuda" if torch.cuda.is_available() else "cpu")
    summary, brief = draft_brief(vlm, Path(silent_video), [(float(a), float(b)) for a, b in segments])
    del vlm
    torch.cuda.empty_cache()
    write_brief(path, brief, video, summary)
    log.info("scene brief: drafted %s (%d entries); edit it and re-run to steer the music", path, len(brief))
    return brief


def setup(cfg, index, brief: list[dict]) -> BriefIndex:
    from bgm.features.hf import ClapEncoder
    import torch
    enc = ClapEncoder(cfg.features.sem_music.clap_model, "cuda" if torch.cuda.is_available() else "cpu")
    t = enc.encode_text([brief_text(e) for e in brief]).astype(np.float64)
    del enc
    return BriefIndex(index, brief, t)


def silent_spans(brief: list[dict], min_s: float) -> list[tuple[float, float]]:
    """Merged [start, end) s of the brief entries without music (needs_music false, or music 'none'), keeping spans of at
    least min_s (shorter ones keep their music: no flicker for a single shot)."""
    spans = []
    for e in brief:
        off = (not e.get("needs_music", True)) or str(e.get("music", "")).strip().lower() in ("none", "no music", "silence")
        if not off:
            continue
        if spans and abs(spans[-1][1] - e["start_s"]) < 0.05:
            spans[-1] = (spans[-1][0], float(e["end_s"]))
        else:
            spans.append((float(e["start_s"]), float(e["end_s"])))
    return [s for s in spans if s[1] - s[0] >= min_s]
