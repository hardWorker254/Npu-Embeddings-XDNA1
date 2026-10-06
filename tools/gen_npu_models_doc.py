#!/usr/bin/env python3
#===----------------------------------------------------------------------===//
# Regenerate NPU_MODELS.md and NPU_MODELS.ru.md.
#
# The companion of tools/gen_npu_ops_doc.py. That one answers "what does an
# ARCHITECTURE support" -- 5 rows, one per kind -- and this one answers the
# question a user actually has when they type a model name: "what can THIS model
# put on the array". 16 rows from tools/data/npu_targets.json, 8 codes, 128
# cells.
#
#   python tools/gen_npu_models_doc.py            # rewrite both files
#   python tools/gen_npu_models_doc.py --check    # exit 1 if either is stale
#   python tools/gen_npu_models_doc.py --only gemma-300m   # one model in full
#
# WHY A SECOND TABLE AND NOT A COLUMN ON THE FIRST
# ----------------------------------------------
# A kind's row is almost always the answer, so most of this document is the
# architecture document repeated seventeen times. The repetition is still worth
# printing, for two reasons, and both are model facts the kind table cannot hold:
#
#   * a model whose ENCODER differs from its kind's -- gemma, and only gemma;
#   * a GATED FFN, whose activation is part of the gated path between ffn_up and
#     ffn_down rather than a standalone pass, so `gelu` is dropped for nomic and
#     gte even though their KIND honours it.
#
# The second one is the whole reason to have this file. `--npu-ops gelu` is
# refused for gemma and silently absent from nomic's and gte's compiled sets, and
# from the architecture table alone a reader would conclude those three models
# behave alike.
#
# EVERY CELL IS DERIVED, NONE IS TYPED
# ------------------------------------
# Statuses come from tools/lib/npu_ops.py's registry through registry_for(), and
# the compiled-set column is npu_ops.buildable_codes() -- the exporter's own
# function, so the "what gets built" column is the exporter's answer and not a
# second opinion about it. The geometry columns re-derive resolve.py's documented
# `_pick` chain (arg > model_spec > defaults > FALLBACK) and import the two
# FALLBACK constants from the exporter rather than restating them, so a change to
# a fallback cannot leave this table quoting the old number.
#===----------------------------------------------------------------------===//

import argparse
import difflib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "lib"))
sys.path.insert(0, str(REPO / "tools" / "export" / "exporters" / "common"))
import npu_ops  # noqa: E402  -- needs the path above
from consts import FALLBACK_HIDDEN, FALLBACK_LN_EPS  # noqa: E402

TARGETS = REPO / "tools" / "data" / "npu_targets.json"
DOCS = {"en": REPO / "NPU_MODELS.md", "ru": REPO / "NPU_MODELS.ru.md"}

# The model-level cell statuses. Four of the five are the registry's own; GATED
# is new and exists only here, because it is a fact about a MODEL and not about a
# kind: the kind says `honours`, and this model has no standalone pass to honour
# it with.
GATED = "gated"

# Kind -> (status, mark en, mark ru). The first four come from the registry and
# their words are fixed there; GATED's two words are this file's.
#
# THE THREE WORDS A CELL CAN CARRY, AND WHY `ABSENT` IS NOW A WORD AND NOT A DASH.
# The table answers one question -- "может ли эта модель отправить эту операцию на
# массив" -- and a reader who has to map "-" onto "нет" is doing a step the table
# could have done for them. A dash in a table also reads as "empty" or "not
# applicable", and `absent` means neither: it means the model HAS no such
# operation, which is a fact about the checkpoint rather than a gap in the table.
# `есть` and `нет` are the two halves of that fact and they are both now said.
#
# `ON_ARRAY` KEPT ITS OWN WORD, and this is the one cell where a three-value
# scheme would have to lie. `on_array` means the work is ALREADY dispatched with no
# code, so `--npu-ops conv` for that model is REFUSED BY NAME -- calling the cell
# `есть` would invite exactly the command that fails, and calling it `невозможно`
# would say the board cannot do it when it is doing it. One cell out of 144 keeps
# a fourth word, and that is cheaper than 143 correct cells plus one misleading.
MARKS = {
    npu_ops.HONOURS: ("**yes**", "**есть**"),
    npu_ops.ON_ARRAY: ("already", "**уже**"),
    npu_ops.UNIMPLEMENTED: ("no code", "**нет**"),
    npu_ops.BLOCKED: ("impossible", "**невозможно**"),
    npu_ops.ABSENT: ("no", "нет"),
    GATED: ("**gated**", "**гейт**"),
}

# Model names in the order the document prints them: the catalogue's own order,
# which groups the text embedders, then whisper, then ViT, then pose. Reordering
# by hand would be a second thing to keep in step with npu_targets.json, and the
# grouping is the useful part anyway.
def models(t):
    return list(t["models"].items())


def kind_of(spec):
    """A text embedder's kind is the default, and saying so is the point.

    `kind: null` is seven of the seventeen rows, and it means "gemm_rtp". Printed as
    an empty cell a reader cannot tell a missing field from a missing kind, so it
    is printed as the default it is.
    """
    return spec.get("kind") or "gemm_rtp"


def is_gated(spec):
    return bool(spec.get("gated_ffn"))


def cell(model, code):
    """The model-level status for one cell."""
    spec = TARGETS_MODEL[model]
    kind = kind_of(spec)
    if is_gated(spec):
        reg = npu_ops.registry_for(kind, model)
        if reg[code][0] == npu_ops.HONOURS and code == "gelu":
            return GATED
    return npu_ops.registry_for(kind, model)[code][0]


def built(model):
    """The sibling design directories the exporter compiles for this model.

    npu_ops.buildable_codes() -- the exporter's own function. Note what it does
    NOT return: the five stream-only codes. `conv` and `attn` are honoured by
    whisper and appear in no compiled directory, because they are streams inside
    the set the GEMM export already produces. A "compiled" column that listed
    them would be wrong, and the header says so in those words.
    """
    spec = TARGETS_MODEL[model]
    return npu_ops.buildable_codes(kind_of(spec), model, is_gated(spec))


def ln_cols(spec, defaults):
    """resolve.py's `_pick(args.ln_cols, layer_norm_cols, hidden, defaults.hidden,
    FALLBACK_HIDDEN)`, with the CLI argument out of scope for a document."""
    for v in (spec.get("layer_norm_cols"), spec.get("hidden"),
              defaults.get("hidden"), FALLBACK_HIDDEN):
        if v is not None:
            return v
    return FALLBACK_HIDDEN


def ln_eps(spec, defaults):
    """resolve.py's `_pick(args.ln_eps, layer_norm_eps, defaults.layer_norm_eps,
    FALLBACK_LN_EPS)`. The fallback is imported, not restated, so a change to the
    exporter's fallback cannot leave this table quoting the old number -- which is
    the failure mode a hand-written geometry column always has."""
    for v in (spec.get("layer_norm_eps"), defaults.get("layer_norm_eps"),
              FALLBACK_LN_EPS):
        if v is not None:
            return v
    return FALLBACK_LN_EPS


def ln_columns(spec, defaults, model):
    """ln_cols / ln_eps for the `layernorm/` design, or None when it is not built.

    Both numbers exist to compile ONE design. A model whose `layn` cell is not
    honoured has no `layernorm/` coming out of it, so printing a row width and an
    epsilon for it is noise that reads as a fact -- yolov8n-pose has no
    `hidden` at all, and its row would have shown a confident 384 that belongs to
    no design anyone will ever build. The fallback chain is still resolve.py's;
    it is simply not consulted for a model that will not use the answer.
    """
    if "layn" not in built(model):
        return None, None
    return ln_cols(spec, defaults), ln_eps(spec, defaults)


def fmt_eps(v, from_model):
    """epsilon, marked when it is the fallback rather than the model's own.

    Seven of the seventeen models carry no `layer_norm_eps`, and the exporter then
    uses FALLBACK_LN_EPS. Printing one number in both cases would make the
    fallback look like a per-model fact.
    """
    return (repr(v) if from_model else f"{v!r} (fallback)")


#== The Russian prose ======================================================#
# Translation lives here for the same reason it lives in gen_npu_ops_doc.py: the
# generators are documentation, the registry is data imported by the exporter.
RU_INTRO = [
    "У рантайма восемь кодов операций, и в `tools/data/npu_targets.json` "
    "{n} моделей: {cells} ячеек.",
    "",
    "Это документ-спутник [NPU_OPS.ru.md](NPU_OPS.ru.md). Тот отвечает на "
    "вопрос «что умеет АРХИТЕКТУРА» — пять строк, по одной на kind. Этот — на "
    "тот вопрос, который реально возникает, когда набирают имя модели: «что "
    "конкретно ЭТА модель может отправить на массив». Английская версия: "
    "[NPU_MODELS.md](NPU_MODELS.md).",
    "",
    "Почему отдельная таблица, а не колонка в первой: строка kind'а почти всегда "
    "и есть ответ, так что бо́льшая часть этого файла повторяет архитектурный. "
    "Повтор всё равно стоит напечатать, и вот почему — есть две модели, ответ "
    "которых отличается от ответа их kind'а:",
    "",
    "- **gemma** — её КОДЕР отличается от kind'а (RMSNorm, GeGLU, `GemmaNpuEncoder` "
    "не читает флаг операции вовсе). Это единственный такой случай.",
    "- **nomic и gte** — у них **гейтед FFN**, а у гейтед FFN активация является "
    "частью гейтед-пути между `ffn_up` и `ffn_down`, а не отдельным проходом. "
    "Их kind говорит `honours` про `gelu`, и для этих двух моделей это не так.",
    "",
    "Второе — единственная настоящая причина завести этот файл: `--npu-ops gelu` "
    "у gemma **отвергается**, а у nomic и gte дизайн `gelu/` просто не "
    "собирается. По архитектурной таблице читатель решил бы, что все три "
    "модели ведут себя одинаково.",
]
# Status -> what the tally line says it means, (en, ru). Keyed on the STATUS, not
# on the rendered mark: an earlier version keyed this on the markdown ("**yes**",
# "**да**"), which meant renaming a mark silently turned every tally line into a
# KeyError or, worse, into a wrong lookup that still found the wrong sentence.
STATUS_GLOSS = {
    npu_ops.HONOURS: (
        "runs on the array today",
        "работает на массиве сегодня"),
    npu_ops.ON_ARRAY: (
        "the work is already dispatched without a code, and the code is refused",
        "работа уже диспатчена без кода, и код отвергается"),
    npu_ops.UNIMPLEMENTED: (
        "the model has the operation, no array branch reaches it",
        "операция у модели есть, ветка на массив не написана"),
    npu_ops.BLOCKED: (
        "cannot be moved on this board; the reason is in NPU_OPS.md",
        "перенести на эту плату нельзя, причина названа в NPU_OPS.ru.md"),
    npu_ops.ABSENT: (
        "the model has no such operation",
        "такой операции у модели нет"),
    GATED: (
        "this model's FFN has no standalone activation pass -- see below",
        "у FFN этой модели нет отдельного прохода активации — см. ниже"),
}

# Why the Russian tally names the STATUS where the English one names the mark.
# Same reason as in gen_npu_ops_doc.py: the Russian line reads better with the
# status word, and it keeps the two documents consistent about which one shows
# what.
RU_GATED_HEAD = "## Чем эти две модели отличаются от своего kind"
RU_GATED_BODY = [
    "### gemma — единственная, у которой отличается КОДЕР",
    "",
    "Остальные пятнадцать берут строку своего kind'а целиком. gemma не берёт: её "
    "строка лежит в `MODEL_REGISTRY`, потому что `GemmaNpuEncoder` не читает "
    "флаг операции вообще — это свойство одного кодера, а не семейства, и "
    "делать из него пятый kind значило бы заявить семейство, которого нет.",
    "",
    "Цена этого решения: `layn` для gemma **заблокирован** — нужен RMSNorm, а "
    "`kernels/layernorm.cc` параметризован только `-DLN_COLS/-DLN_EPS/"
    "-DLN_ROWS`, то есть это другое тело ядра и **девятый** код. Намеренно не "
    "сделано: набор кодов стал бы больше и менее однородным. Подробности — в "
    "ячейке `embeddinggemma-300m/layn` файла NPU_OPS.ru.md.",
    "",
    "Итог по gemma: `buildable_codes()` возвращает **пустое множество** — "
    "экспортёр не компилирует для неё ни одного соседнего дизайна.",
    "",
    "### nomic и gte — ГЕЙТЕД FFN, и из-за этого нет `gelu`",
    "",
    "У GeGLU активация считается внутри гейтед-пути, пока умножаются две его "
    "половины, а не отдельным проходом по готовому тензору. Дизайн `gelu/` — это "
    "один `hw_context` с поэлементным проходом, то есть переносить нечего: "
    "исполнение кода напечатало бы «на МАССИВЕ», не сдвинув ничего.",
    "",
    "Именно поэтому в таблице их `gelu` отмечен `**гейт**`, а не `**да**`, хотя "
    "kind `gemm_rtp` отвечает `honours`: ячейка описывает модель, а не вид "
    "сети. Это единственное место, где таблица моделей расходится с таблицей "
    "архитектур, и оно не расходится **по факту**, а по причине.",
]
RU_BUILT_HEAD = "## Что экспортёр компилирует для каждой модели"
RU_BUILT_NOTE = [
    "Эта колонка — ответ самого экспортёра: `npu_ops.buildable_codes()` из "
    "`tools/lib/npu_ops.py`, та же функция, которую зовёт сборка. Не "
    "пересказ.",
    "",
    "Обратите внимание, чего в ней **нет**: пяти кодов без собственного "
    "каталога. `conv` и `attn` у whisper честно работают на массиве (статус "
    "`honours` в матрице выше), но в компилируемый список не попадают, потому "
    "что это потоки **внутри** набора, который экспорт GEMM и так производит. "
    "Колонка, которая перечислила бы их, была бы неверной.",
    "",
    "`ln_cols` и `ln_eps` — числа, которые экспортёр берёт из модели для "
    "дизайна `layernorm/`: ширина строки из `hidden`, а epsilon из "
    "`layer_norm_eps`. У семи моделей этого поля нет, и тогда экспортёр "
    "подставляет `FALLBACK_LN_EPS` — это отмечено словом `(fallback)`, чтобы "
    "значение по умолчанию не выглядело свойством модели. Прочерк в этих "
    "колонках означает, что `layernorm/` для модели не собирается, и числам "
    "некому принадлежать: у `yolov8n-pose` нет даже `hidden`. `vocab` — "
    "ширина словаря: она решает, разбивается ли проекция в логиты на чанки.",
]
RU_WHY_HEAD = "## Почему шестнадцать строк и четыре kind"
RU_WHY_BODY = [
    "kind'ов в `npu_targets.json` четыре (`gemm_rtp`, `stt`, `cls`, `pose`), а "
    "моделей шестнадцать, и таблица по моделям длиннее в четыре раза. Это "
    "осознанно: kind — это то, что можно объединить, а читателю нужен ответ про "
    "имя, которое он вводит.",
    "",
    "Раскладка: 7 текстовых эмбеддеров без своего `kind` (то есть `gemm_rtp`, "
    "и это отмечено в колонке `kind`), 6 whisper (`stt`), 1 ViT (`cls`), 1 "
    "YOLO-pose (`pose`).",
    "",
    "Ни одна строка здесь не выдумана: список моделей читается из "
    "`tools/data/npu_targets.json`, а не поддерживается вручную, поэтому "
    "модель, добавленная в каталог, попадёт в таблицу сама.",
]
RU_CHECKED_HEAD = "## Где это проверяется"
RU_CHECKED = [
    "- `tools/data/npu_targets.json` — список моделей и их `kind` / "
    "`gated_ffn` / геометрия.",
    "- `tools/lib/npu_ops.py` — реестр; статусы и `buildable_codes()`.",
    "- `tools/gen_npu_models_doc.py` — этот файл целиком.",
    "- `tools/verify/verify_npu_op_matrix.py` — проверяет, что оба файла "
    "ровно то, что печатает генератор, и сверяет сводку по статусам с той, что "
    "в реестре.",
]

TARGETS_MODEL = {}
TARGETS_DATA = {}


def tally():
    counts = {s: 0 for s in npu_ops.STATUSES}
    counts[GATED] = 0
    for model, _spec in models(TARGETS_DATA):
        for code in npu_ops.OPS:
            counts[cell(model, code)] += 1
    return counts


def matrix_table(ru=False):
    """One row per model, listing the codes that model can move -- NOT a grid.

    WHY NOT A GRID, and this is a reversal rather than a first choice. The grid
    was models x codes, 144 cells, and 55 of them said the model has no such
    operation -- more than a third of the table carrying the same absence in the
    same position. Reading it means filtering the empty cells out on every row to
    find the answer, and the filtering IS the answer.

    A row that lists what IS there has no empty cell to interpret, so there is
    nothing to map a dash onto and nothing that can be misread as "empty" or "not
    applicable". The absence is carried by the header instead: the columns say what
    the lists mean, and a code that is not listed is not something the model has.
    That is the same fact, said once instead of 55 times.

    WHAT THE COLUMNS ARE, because a list that silently dropped the awkward cases
    would be the easy way to lose them:
      * `есть` / `yes`  -- dispatches today
      * `невозможно` / `impossible` -- the model HAS the code and the board cannot
        take it; the reason is in the architecture row, and dropping it here would
        make a refusal invisible
      * `уже` / `already` -- already dispatched with no code, so the code is
        refused BY NAME; this is the one cell where a two-list table would lie
      * `гейт` / `gated` -- dispatches, and the operation is folded into a gated
        path so there is nothing separate for it to take over. nomic and gte are
        the two, and their kind says `honours` while the truth for those two models
        is not that.
    """
    # ONE column, and every entry in it names its own status.
    #
    # The shapes this went through, because the count of empty cells is the
    # argument: a grid had 55 "no" cells and 144 cells total; three columns cut
    # the absences to 54 dashes, and the three columns for the three awkward
    # statuses had to be interpreted per row; one column leaves NONE, and an
    # entry like `gelu` (гейт) carries its own caveat to the reader instead of
    # relying on a heading three columns to the left.
    #
    # The caveat is a parenthetical and not a second column because a status that
    # needs a separate column is a status the table would rather not have: `уже`
    # and `невозможно` are rarer than `есть`, and putting them inline says that.
    suffix = {npu_ops.BLOCKED: "невозможно" if ru else "impossible",
              npu_ops.ON_ARRAY: "уже" if ru else "already",
              GATED: "гейт" if ru else "gated"}
    head = ("| модель | что уходит на массив |" if ru else
            "| model | what goes to the array |")
    lines = [head, "| --- | --- |"]
    for model, spec in models(TARGETS_DATA):
        items = []
        for code in npu_ops.OPS:
            st = cell(model, code)
            if st == npu_ops.HONOURS:
                items.append(f"`{code}`")
            elif st in suffix:
                items.append(f"`{code}` ({suffix[st]})")
        cells = ", ".join(items) or ("_ничего_" if ru else "_nothing_")
        lines.append(f"| `{model}` | {cells} |")
    return lines


def built_table(ru=False):
    defaults = TARGETS_DATA["defaults"]
    if ru:
        head = ("| модель | kind | что компилируется | ln_cols | ln_eps | vocab |")
    else:
        head = "| model | kind | compiled designs | ln_cols | ln_eps | vocab |"
    lines = [head, "| --- | --- | --- | --- | --- | --- |"]
    for model, spec in models(TARGETS_DATA):
        dirs = built(model)
        built_s = ("`" + "`, `".join(sorted(dirs)) + "`") if dirs else \
            ("_none_" if not ru else "_ничего_")
        cols, eps = ln_columns(spec, defaults, model)
        vocab = spec.get("vocab")
        lines.append(
            f"| `{model}` | `{kind_of(spec)}` | {built_s} | "
            f"{cols if cols is not None else '—'} | "
            f"{fmt_eps(eps, 'layer_norm_eps' in spec) if eps is not None else '—'} | "
            f"{vocab if vocab is not None else '—'} |")
    return lines


def doc(ru=False):
    idx = 1 if ru else 0
    n_models = len(models(TARGETS_DATA))
    counts = tally()
    n_cells = n_models * len(npu_ops.OPS)

    L = []
    a = L.append
    if ru:
        a("# Какую операцию каждая модель может отправить на массив")
        a("")
        a("<!-- СГЕНЕРИРОВАНО tools/gen_npu_models_doc.py из "
          "tools/data/npu_targets.json и tools/lib/npu_ops.py. Не править "
          "руками: запустите скрипт, иначе упадёт "
          "tools/verify/verify_npu_op_matrix.py. Английская версия: "
          "NPU_MODELS.md -->")
        a("")
        for line in RU_INTRO:
            a(line.format(n=n_models, cells=n_cells) if "{n}" in line else line)
        a("")
        a(f"Итог по {n_cells} ячейкам:")
        a("")
        for s in list(npu_ops.STATUSES) + [GATED]:
            a(f"- **{counts[s]}** `{s}` — {STATUS_GLOSS[s][1]};")
        a("")
        a("## Что каждая модель может отправить на массив")
        a("")
        a("Не сетка моделей на операции, а список на модель: в сетке из "
          f"{n_cells} ячеек {counts[npu_ops.ABSENT]} говорили бы одно и то же "
          "отсутствие, и читать ответ пришлось бы выбиранием пустых ячеек по "
          "всей строке. Выборка и есть ответ, поэтому в таблице ни одной пустой "
          "ячейки нет, а отсутствие несёт заголовок: код, которого в списке не "
          "нет, — такой операции у модели нет.")
        a("")
        L.extend(matrix_table(ru=True))
        a("")
        a(RU_GATED_HEAD)
        a("")
        L.extend(RU_GATED_BODY)
        a("")
        a(RU_BUILT_HEAD)
        a("")
        L.extend(RU_BUILT_NOTE)
        a("")
        L.extend(built_table(ru=True))
        a("")
        a(RU_WHY_HEAD)
        a("")
        L.extend(RU_WHY_BODY)
        a("")
        a(RU_CHECKED_HEAD)
        a("")
        L.extend(RU_CHECKED)
        a("")
        return "\n".join(L)

    a("# Which op each model can put on the array")
    a("")
    a("<!-- GENERATED by tools/gen_npu_models_doc.py from")
    a("     tools/data/npu_targets.json and tools/lib/npu_ops.py. Do not edit:")
    a("     run the script, or tools/verify/verify_npu_op_matrix.py fails.")
    a("     Russian version: NPU_MODELS.ru.md -->")
    a("")
    a(f"Eight op codes, {n_models} models in `tools/data/npu_targets.json`, "
      f"{n_cells} cells.")
    a("")
    a("This is the companion to [NPU_OPS.md](NPU_OPS.md), which answers \"what "
      "does an ARCHITECTURE support\" -- five rows, one per kind. This one "
      "answers the question you actually have when you type a model name: "
      "\"what can THIS model put on the array\". Russian: "
      "[NPU_MODELS.ru.md](NPU_MODELS.ru.md).")
    a("")
    a("Why a second table rather than a column on the first: a kind's row is "
      "almost always the answer, so most of this file repeats the architecture "
      "document seventeen times. The repetition is still worth printing, because "
      "two models do NOT answer with their kind's row:")
    a("")
    a("- **gemma** -- its ENCODER differs from its kind's (RMSNorm, GeGLU, and "
      "`GemmaNpuEncoder` reads no per-op flag at all). It is the only one.")
    a("- **nomic and gte** -- they have a **gated FFN**, whose activation is "
      "part of the gated path between `ffn_up` and `ffn_down` rather than a "
      "standalone pass. Their kind says `honours` for `gelu`, and for these two "
      "models that is not true.")
    a("")
    a("The second is the actual reason this file exists: `--npu-ops gelu` is "
      "**refused** for gemma, and for nomic and gte no `gelu/` design is "
      "compiled at all. From the architecture table alone a reader would "
      "conclude those three behave alike.")
    a("")
    a(f"Tally over the {n_cells} cells:")
    a("")
    for s in list(npu_ops.STATUSES) + [GATED]:
        a(f"- **{counts[s]}** {MARKS[s][0]} -- {STATUS_GLOSS[s][0]}")
    a("")
    a("## What each model can send to the array")
    a("")
    a("A list per model, not a grid of models against operations. In a "
      f"{n_cells}-cell grid, {counts[npu_ops.ABSENT]} of the cells would say "
      "the same absence in the same position, and reading the answer meant "
      "filtering the empty ones out of every row -- and the filtering IS the "
      "answer. There are no empty cells here at all: one column, and every entry "
      "in it names its own status, so `gelu` (gated) reads without a heading to "
      "its left. The header carries the absence: a code that is not listed is not "
      "an operation this model has.")
    a("")
    L.extend(matrix_table())
    a("")
    a("## Where these two models differ from their kind")
    a("")
    L.extend(EN_GATED_BODY)
    a("")
    a("## What the exporter compiles for each model")
    a("")
    L.extend(EN_BUILT_NOTE)
    a("")
    L.extend(built_table())
    a("")
    a("## Why seventeen rows and five kinds")
    a("")
    L.extend(EN_WHY_BODY)
    a("")
    a("## Where this is checked")
    a("")
    L.extend(EN_CHECKED)
    a("")
    return "\n".join(L)


EN_GATED_BODY = [
    "### gemma -- the only model whose ENCODER differs",
    "",
    "The other sixteen take their kind's row wholesale. gemma does not: its row "
    "lives in `MODEL_REGISTRY`, because `GemmaNpuEncoder` reads no per-op flag "
    "at all. That is a property of one encoder and not of a family, and making "
    "it another kind would claim a family that does not exist.",
    "",
    "The price of that decision is visible in the matrix: gemma's `layn` is "
    "**blocked** -- it needs RMSNorm, and `kernels/layernorm.cc` is "
    "parameterised only by `-DLN_COLS/-DLN_EPS/-DLN_ROWS`, so this is a "
    "different kernel body and a **ninth** code. Deliberately not done: the code "
    "set would grow and lose its uniformity. The full argument is the "
    "`embeddinggemma-300m/layn` cell in NPU_OPS.md.",
    "",
    "So `buildable_codes()` returns the **empty set** for gemma: the exporter "
    "compiles no sibling design at all for it.",
    "",
    "### nomic and gte -- a GATED FFN, and therefore no `gelu`",
    "",
    "In GeGLU the activation is computed inside the gated path while its two "
    "halves are multiplied, not as a standalone pass over a finished tensor. "
    "The whole `gelu/` design is one `hw_context` doing an elementwise pass, so "
    "there is nothing to move: honouring the code would print \"on the ARRAY\" "
    "while moving nothing.",
    "",
    "That is why their `gelu` is marked `**gated**` here rather than `**yes**`, "
    "even though kind `gemm_rtp` answers `honours`: the cell describes a model, "
    "not a network shape. It is the only place this table and the architecture "
    "table disagree, and they disagree about the **reason**, not the outcome.",
]

EN_BUILT_NOTE = [
    "This column is the exporter's own answer: `npu_ops.buildable_codes()` out "
    "of `tools/lib/npu_ops.py`, the same function the build calls. Not a "
    "second opinion about it.",
    "",
    "Note what is **absent** from it: the five codes with no design directory of "
    "their own. `conv` and `attn` genuinely run on the array for whisper -- the "
    "status is `honours` in the matrix above -- and neither appears here, "
    "because they are streams **inside** the set the GEMM export already "
    "produces. A column that listed them would be wrong.",
    "",
    "`ln_cols` and `ln_eps` are the numbers the exporter takes from the model "
    "for the `layernorm/` design: row width from `hidden`, epsilon from "
    "`layer_norm_eps`. Seven models carry no such field, and the exporter then "
    "substitutes `FALLBACK_LN_EPS` -- marked `(fallback)` here so a default does "
    "not read as a per-model fact. A dash in those columns means no "
    "`layernorm/` is compiled for that model, so the numbers belong to nothing: "
    "`yolov8n-pose` does not even carry a `hidden`. `vocab` is the vocabulary "
    "width, which decides whether the logit projection is chunked.",
]

EN_WHY_BODY = [
    "`npu_targets.json` knows five kinds (`gemm_rtp`, `stt`, `cls`, `pose`, "
    "`hands`) and seventeen models, so the model table is a little under four "
    "times the length of the kind table. That is deliberate: a kind is what can "
    "be shared, and the reader wants the answer for the name they typed.",
    "",
    "The split: 7 text embedders with no `kind` of their own (so `gemm_rtp`, and "
    "the `kind` column says so rather than leaving a blank), 6 whisper (`stt`), "
    "one ViT (`cls`), one YOLO-pose (`pose`), one MediaPipe hands "
    "(`mediapipe-hands`).",
    "",
    "No row here is typed by hand. The model list is read from "
    "`tools/data/npu_targets.json`, so a model added to the catalogue lands in "
    "this table by itself -- which is the point of generating it.",
]

EN_CHECKED = [
    "- `tools/data/npu_targets.json` -- the model list and each model's `kind`, "
    "`gated_ffn` and geometry.",
    "- `tools/lib/npu_ops.py` -- the registry, and `buildable_codes()`.",
    "- `tools/gen_npu_models_doc.py` -- this file, in full.",
    "- `tools/verify/verify_npu_op_matrix.py` -- asserts both files are exactly "
    "what the generators print, and that the status tally matches the registry's.",
]


def load():
    global TARGETS_DATA, TARGETS_MODEL
    TARGETS_DATA = json.loads(TARGETS.read_text())
    TARGETS_MODEL = TARGETS_DATA["models"]


def check_one(path, text) -> bool:
    if not path.exists():
        print(f"{path.name} does not exist; run tools/gen_npu_models_doc.py")
        return False
    cur = path.read_text()
    if cur == text:
        print(f"ok    {path.name} matches npu_targets.json and the registry")
        return True
    print(f"{path.name} is stale; tools/gen_npu_models_doc.py prints:")
    for line in difflib.unified_diff(cur.splitlines(), text.splitlines(),
                                     path.name, "generated", lineterm="", n=1):
        print("  " + line)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 and print the diff if stale")
    ap.add_argument("--only", metavar="MODEL",
                    help="print one model's section instead of the whole "
                         "document; a lookup, not a generator mode")
    ns = ap.parse_args()
    load()
    if ns.only:
        spec = TARGETS_MODEL.get(ns.only)
        if spec is None:
            print(f"{ns.only} is not in npu_targets.json; known: "
                  f"{', '.join(TARGETS_MODEL)}")
            return 1
        kind = kind_of(spec)
        print(f"{ns.only}  kind={kind}  gated={is_gated(spec)}")
        for code in npu_ops.OPS:
            st = cell(ns.only, code)
            print(f"  {code:6s} {st:14s} {MARKS[st][0]}")
        print(f"  compiled: {sorted(built(ns.only)) or 'nothing'}")
        return 0
    texts = {"en": doc(False), "ru": doc(True)}
    if ns.check:
        ok = all(check_one(DOCS[k], texts[k]) for k in ("en", "ru"))
        return 0 if ok else 1
    for k in ("en", "ru"):
        DOCS[k].write_text(texts[k])
        print(f"wrote {DOCS[k].name} ({len(texts[k].splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())