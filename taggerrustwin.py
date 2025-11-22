# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, re, unicodedata, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# --- пробуем подключить Rust-движок ---
try:
    from rust_ac_tagger import RustAC  # класс из lib.rs
except Exception:
    RustAC = None

# -------- настройки путей --------
BASE_DIR   = Path(r"C:\Experiments\project\corpus\AA")
OUTPUT_DIR = BASE_DIR.parent / "AA_tagged"
MERGED_PATH = Path(__file__).parent / "merged.json"
MASK = "wiki_*.txt"

# -------- приоритеты --------
PRIORITY: Dict[str, int] = {
    "<|intellect|>": 0,"<|emotion|>": 1,
    "<|movement|>": 2, "<|instinct|>": 3,
    "<|negativeemotion|>": 4
}
DEFAULT_PRIORITY = 9

# -------- служебные регексы/утилиты --------
SKIP_LINE_BOL = re.compile(r"^\s*(?:<\|doc\|>|<\|title\|>|<\|section\|>|<\|subsection\|>|<\|endofdoc\|>)\b")
TOKEN_RE = re.compile(r"<\|[^|]+?\|>")
WORD_CHARS = re.compile(r"[0-9A-Za-zА-Яа-яЁё\-]")

def normalize_for_match(s: str) -> str:
    x = unicodedata.normalize("NFC", s)
    x = x.replace("ё", "е").replace("Ё", "Е")
    return x.lower()

def is_word_char(ch: str) -> bool:
    return bool(ch and WORD_CHARS.match(ch))

def at_word_boundaries(s: str, start: int, end: int) -> bool:
    left_ok  = (start == 0) or (not is_word_char(s[start - 1]))
    right_ok = (end   == len(s)) or (not is_word_char(s[end]))
    return left_ok and right_ok

def strip_tokens_build_map(raw: str) -> Tuple[str, List[int]]:
    shadow, s2r, i = [], [], 0
    while i < len(raw):
        m = TOKEN_RE.match(raw, i)
        if m:
            i = m.end(); continue
        shadow.append(raw[i]); s2r.append(i); i += 1
    return "".join(shadow), s2r

def map_span_to_raw(s_start: int, s_end: int, s2r: List[int]) -> Tuple[int, int]:
    """
    Безопасная версия маппинга индексов shadow->raw.
    Проверяет границы и, при несоответствии, выбрасывает понятную ошибку.
    """
    if s_start < 0 or s_end <= s_start:
        raise ValueError(f"Invalid shadow span: start={s_start}, end={s_end}")
    if not s2r:
        raise ValueError("s2r mapping is empty")

    # s2r индексируется по позициям в shadow: допустимые индексы 0..len(s2r)-1
    if s_start >= len(s2r):
        raise IndexError(f"s_start {s_start} >= len(s2r) {len(s2r)}")
    if (s_end - 1) >= len(s2r):
        raise IndexError(f"s_end-1 {s_end-1} >= len(s2r) {len(s2r)}")

    raw_start = s2r[s_start]
    raw_end = s2r[s_end - 1] + 1
    return raw_start, raw_end

# -------- числа <|num|> --------
THIN_SPACES = "\u00A0\u2009\u202F"
NUM_CORE = rf"(?:\d{{1,3}}(?:[ {THIN_SPACES}]?\d{{3}})+|\d+)(?:[.,]\d+)?"
NUM_PCT  = r"(?:\s?[%‰])?"
ORD_SFX  = r"(?:-?(?:й|я|е|го|ого|ему|му|ми|х))?"
NUM_RE = re.compile(rf"{NUM_CORE}{NUM_PCT}{ORD_SFX}")

def find_numbers_in_shadow(shadow: str) -> List[Tuple[int, int]]:
    spans = []
    for m in NUM_RE.finditer(shadow):
        s, e = m.span()
        if at_word_boundaries(shadow, s, e):
            spans.append((s, e))
    return spans

# -------- математика <|math|> --------
ROMAN_RE = re.compile(r"(?<!\w)(M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{1,3}))(?!\w)", re.IGNORECASE)
UNICODE_FRAC_RE = re.compile(r"[\u00BC-\u00BE\u2150-\u215E]")
SLASH_FRAC_RE   = re.compile(r"(?<!\w)\d+\s*/\s*\d+(?!\w)")
MIXED_FRAC_RE   = re.compile(r"(?<!\w)\d+\s+\d+\s*/\s*\d+(?!\w)")
OPS_RE = re.compile(r"(?<!\w)(?:[+\-±∓×xX*⋅·÷/=<>≤≥≠≈^~]+)(?!\w)")
GREEK_SYM_RE = re.compile(r"(?<!\w)[αβγδΔθλμπΠφΦΩωΣσΤτ](?!\w)")
SUPSUB_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎]")

def _add_span(spans: List[Tuple[int, int]], s: int, e: int):
    if s < e: spans.append((s, e))

def find_math_in_shadow(shadow: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for m in MIXED_FRAC_RE.finditer(shadow):
        _add_span(spans, *m.span())
    for m in SLASH_FRAC_RE.finditer(shadow):
        _add_span(spans, *m.span())
    for m in UNICODE_FRAC_RE.finditer(shadow):
        _add_span(spans, *m.span())
    for m in ROMAN_RE.finditer(shadow):
        s, e = m.span()
        if e - s >= 2: _add_span(spans, s, e)
    for m in GREEK_SYM_RE.finditer(shadow):
        _add_span(spans, *m.span())
    for m in OPS_RE.finditer(shadow):
        _add_span(spans, *m.span())
    for m in SUPSUB_RE.finditer(shadow):
        _add_span(spans, *m.span())
    return spans

# -------- merged.json → индекс --------
def load_merged(path: Path) -> Dict[str, Dict[str, List[str]]]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return {k: v for k, v in data.items()
            if k.startswith("<|") and k.endswith("|>") and isinstance(v, dict)}

def build_index(merged_path: Path, min_len: int = 3) -> Tuple[List[str], List[List[Tuple[str,int]]]]:
    merged = load_merged(merged_path)
    form2tags: Dict[str, set] = {}
    for tag, bucket in merged.items():
        for term, forms in (bucket.items() if isinstance(bucket, dict) else []):
            if not isinstance(forms, list): continue
            for f in forms:
                if not isinstance(f, str): continue
                fx = f.strip()
                if not fx: continue
                nf = normalize_for_match(fx)
                if len(nf) < min_len:
                    continue
                form2tags.setdefault(nf, set()).add(tag)
    forms = list(form2tags.keys())
    tags_per_form: List[List[Tuple[str,int]]] = []
    for f in forms:
        lst = sorted(((t, PRIORITY.get(t, DEFAULT_PRIORITY)) for t in form2tags[f]),
                     key=lambda x: (x[1], x[0]))
        tags_per_form.append(lst)
    return forms, tags_per_form

def insert_nested_tags(s: str, spans: List[Tuple[int,int,str,int]]) -> str:
    opens, closes = {}, {}
    for st, en, tg, pr in spans:
        opens.setdefault(st, []).append((pr, tg))
        closes.setdefault(en, []).append((pr, tg))
    out = []
    for i, ch in enumerate(s):
        if i in opens:
            for pr, tg in sorted(opens[i], key=lambda x: x[0]):
                out.append(tg)
        out.append(ch)
        if i + 1 in closes:
            for pr, tg in sorted(closes[i + 1], key=lambda x: -x[0]):
                out.append(tg)
    if len(s) in closes:
        for pr, tg in sorted(closes[len(s)], key=lambda x: -x[0]):
            out.append(tg)
    return "".join(out)

# -------- NER (Natasha) --------
def init_ner():
    try:
        from natasha import Doc, NewsNER, NewsEmbedding, Segmenter
        return {
            "Doc": Doc,
            "Segmenter": Segmenter(),
            "NER": NewsNER(NewsEmbedding())
        }
    except Exception:
        return None

def ner_protect_spans(shadow: str, ner) -> List[Tuple[int,int]]:
    if ner is None:
        return []
    doc = ner["Doc"](shadow)
    doc.segment(ner["Segmenter"])
    ner["NER"](doc)
    out = []
    for span in doc.spans:
        if span.type in ("PER", "LOC"):
            s, e = span.start, span.stop
            if 0 <= s < e <= len(shadow):
                out.append((s, e))
    return out

def overlaps(s1:int,e1:int,s2:int,e2:int)->bool:
    return not (e1 <= s2 or e2 <= s1)

def tag_line(
    line: str,
    forms: List[str],
    tags_per_form: List[List[Tuple[str,int]]],
    ner,
    eng=None
) -> str:
    if SKIP_LINE_BOL.match(line):
        return line
    src_raw = line
    shadow, s2r = strip_tokens_build_map(line)
    shadow_norm = normalize_for_match(shadow)
    ner_spans = ner_protect_spans(shadow, ner)

    spans: List[Tuple[int,int,str,int]] = []

    # 1) словарные формы: Rust-движок, если доступен; иначе — старый Python-режим
    if eng is not None:
        try:
            matches = eng.find_all(shadow_norm)
        except Exception:
            matches = []
        for s, e, wid in matches:
            if not (0 <= wid < len(tags_per_form)):
                continue
            if any(overlaps(s, e, ps, pe) for ps, pe in ner_spans):
                continue
            # доп. контроль границ слова (как раньше через \b)
            if not at_word_boundaries(shadow_norm, s, e):
                continue
            rs, re_ = map_span_to_raw(s, e, s2r)
            for tag, pr in tags_per_form[wid]:
                spans.append((rs, re_, tag, pr))
    else:
        for wid, w in enumerate(forms):
            pattern = re.compile(rf"\b{re.escape(w)}\b")
            for m in pattern.finditer(shadow_norm):
                s, e = m.start(), m.end()
                if any(overlaps(s, e, ps, pe) for ps, pe in ner_spans):
                    continue
                rs, re_ = map_span_to_raw(s, e, s2r)
                for tag, pr in tags_per_form[wid]:
                    spans.append((rs, re_, tag, pr))

    # 2) числа
    for s, e in find_numbers_in_shadow(shadow):
        if any(overlaps(s, e, ps, pe) for ps, pe in ner_spans):
            continue
        rs, re_ = map_span_to_raw(s, e, s2r)
        spans.append((rs, re_, "<|num|>", PRIORITY.get("<|num|>", DEFAULT_PRIORITY)))

    # 3) математика
    for s, e in find_math_in_shadow(shadow):
        if any(overlaps(s, e, ps, pe) for ps, pe in ner_spans):
            continue
        rs, re_ = map_span_to_raw(s, e, s2r)
        spans.append((rs, re_, "<|math|>", PRIORITY.get("<|math|>", DEFAULT_PRIORITY)))

    if not spans:
        return src_raw
    return insert_nested_tags(src_raw, spans)

# ---------- ПАРАЛЛЕЛЬ НАД ОДНИМ ФАЙЛОМ (4 ядра) ----------
def _process_chunk(
    src: Path,
    chunk_idx: int,
    start_line: int,
    end_line: int,
    tmp_out: Path,
    forms: List[str],
    tags_per_form: List[List[Tuple[str,int]]],
    position: int = 0,
) -> Tuple[int, int]:
    ner = init_ner()

    # локальный Rust-движок на воркера (если библиотека есть)
    eng = None
    if RustAC is not None:
        try:
            eng = RustAC(forms)
        except Exception:
            eng = None

    # читаем только нужный диапазон строк
    with src.open("r", encoding="utf-8", errors="replace") as fin, \
         tmp_out.open("w", encoding="utf-8") as fout:
        # быстрый пропуск
        for _ in range(start_line):
            fin.readline()
        total = end_line - start_line
        iterator = range(total)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=total,
                unit="lines",
                desc=f"{src.name} [{chunk_idx}]",
                dynamic_ncols=True,
                position=position,
                leave=False,
            )
        line_no = start_line  # текущий номер строки в исходном файле (0-based)
        for _ in iterator:
            line = fin.readline()
            if not line:
                break
            try:
                processed = tag_line(line.rstrip("\n"), forms, tags_per_form, ner, eng=eng)
                fout.write(processed + "\n")
            except Exception as exc:
                # логируем исключение в отдельный файл внутри tmp_dir (рядом с tmp_out)
                err_log = tmp_out.with_name(f"{tmp_out.stem}_errors.txt")
                try:
                    with err_log.open("a", encoding="utf-8") as le:
                        le.write(f"=== Exception in chunk {chunk_idx} line {line_no} ===\n")
                        le.write(f"{type(exc).__name__}: {exc}\n")
                        # сохраняем проблемную строку (без лишних переводов)
                        le.write(f"LINE: {line.rstrip(chr(10))!r}\n\n")
                except Exception:
                    # если логирование само падает — тихо молчим, но продолжаем
                    pass
                # записываем оригинальную строку, чтобы выходной файл был согласован
                try:
                    fout.write(line)
                except Exception:
                    # в крайне редком случае, если запись не удалась, продолжаем
                    pass
            finally:
                line_no += 1
        if tqdm is not None and hasattr(iterator, "close"):
            iterator.close()
    return (chunk_idx, total)

def process_single_file_parallel(
    src: Path,
    dst: Path,
    forms: List[str],
    tags_per_form: List[List[Tuple[str,int]]],
    workers: int = 4
) -> None:
    # считаем число строк
    with src.open("r", encoding="utf-8", errors="replace") as fin:
        total = sum(1 for _ in fin)

    # границы кусков
    workers = max(1, workers)
    parts = []
    base = total // workers
    rem  = total % workers
    start = 0
    for i in range(workers):
        size = base + (1 if i < rem else 0)
        end = start + size
        parts.append((i, start, end))
        start = end

    tmp_dir = dst.parent / f".tmp_{src.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = []
        for i, s, e in parts:
            tmp_out = tmp_dir / f"part_{i:02d}.txt"
            futs.append(
                ex.submit(
                    _process_chunk,
                    src, i, s, e, tmp_out,
                    forms, tags_per_form, i
                )
            )
        for fut in as_completed(futs):
            fut.result()

    # склейка
    with dst.open("w", encoding="utf-8") as fout:
        for i, _, _ in parts:
            p = tmp_dir / f"part_{i:02d}.txt"
            with p.open("r", encoding="utf-8", errors="replace") as f:
                shutil.copyfileobj(f, fout)

    shutil.rmtree(tmp_dir, ignore_errors=True)

# -------- CLI --------
def main(file_path: Path, workers: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    forms, tags_per_form = build_index(MERGED_PATH, min_len=3)

    # нормализуем путь
    src = file_path if file_path.is_absolute() else (BASE_DIR / file_path)
    if not src.exists():
        raise SystemExit(f"Нет файла: {src}")
    dst = OUTPUT_DIR / src.name

    print(f"Файл: {src.name} | Воркеров: {workers} | RustAC: {'on' if RustAC is not None else 'off'}")
    process_single_file_parallel(src, dst, forms, tags_per_form, workers=workers)
    print(f"✓ {src.name} → {dst} (готово)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Один файл (имя в BASE_DIR или абсолютный путь).")
    parser.add_argument("--workers", type=int, default=4, help="Число процессов-воркеров (по умолчанию 4).")
    args = parser.parse_args()

    p = Path(args.input)
    w = args.workers if args.workers and args.workers > 0 else 4
    if w > 64:  # предохранитель
        w = 64
    main(p, w)
