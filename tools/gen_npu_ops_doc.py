#!/usr/bin/env python3
#===----------------------------------------------------------------------===//
# Regenerate NPU_OPS.md and NPU_OPS.ru.md from tools/lib/npu_ops.py.
#
# The 48 cells (6 architectures x 8 codes) live in the registry as Python because
# the exporter reads them to decide what to compile. A hand-written Markdown copy
# of the same table would be a third copy of the truth and would drift the first
# time somebody added a code, so both documents are GENERATED and
# tools/verify/verify_npu_op_matrix.py asserts the checked-in files are exactly
# what this script prints.
#
#   python tools/gen_npu_ops_doc.py            # rewrite both files
#   python tools/gen_npu_ops_doc.py --check    # exit 1 if either is stale
#
# WHAT IS GENERATED AND WHAT IS TRANSLATED
# ----------------------------------------
# The STATUS of every cell comes from the registry, so the document cannot
# flatter a model: only the exporter's idea of what exists can change a tick, and
# changing that is a behaviour change the matrix gate runs against the binary.
#
# The REASON for every cell is prose that lives in the registry, in English. The
# Russian file carries a translation of each one, kept in REASONS_RU below rather
# than in the registry, because the registry is imported by the exporter and has
# no business carrying two languages: a cell's status is data and its
# explanation is documentation, and they have different lifetimes.
#
# The consequence is that the two files can drift from EACH OTHER -- a reason
# edited in the registry and not translated here leaves NPU_OPS.ru.md stale in a
# way `--check` cannot see, because it only knows the generated text is stable.
# So translation completeness IS checked instead: every one of the 48 cells must
# have a Russian reason or the generator raises, and a stale translation is caught
# by the same gate that catches a stale document.
#===----------------------------------------------------------------------===//

import argparse
import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "lib"))
import npu_ops  # noqa: E402  -- needs the path above

DOCS = {"en": REPO / "NPU_OPS.md", "ru": REPO / "NPU_OPS.ru.md"}

# The order the architectures are printed in. Deliberately NOT the registry's
# dict order: this is the order a reader meets them in -- the default first, then
# the two that have audio, then the two that are not transformers at all.
# (label, title en, title ru, blurb en, blurb ru)
ARCHES = [
    ("gemm_rtp", "Text embedders", "Текстовые эмбеддеры",
     "BERT and friends: bge, MiniLM, nomic, gte. This is also the default for a "
     "text embedder with no `kind` of its own.",
     "BERT и родственные: bge, MiniLM, nomic, gte. Это же значение по "
     "умолчанию для текстового эмбеддера, у которого нет своего `kind`."),
    ("stt", "Speech to text", "Распознавание речи",
     "Whisper (all sizes) and anything else that carries an audio front end.",
     "Whisper (все размеры) и всё, что несёт аудио-фронтенд."),
    ("cls", "Image classification", "Классификация изображений",
     "ViT. Its four GEMM streams are `gemm_rtp`'s, which is why the "
     "classification head lives here.",
     "ViT. Его четыре GEMM-потока — это потоки `gemm_rtp`, и поэтому "
     "классификационная голова живёт здесь."),
    ("pose", "Pose", "Поза",
     "YOLO pose. No transformer, no normalisation, no attention -- most of this "
     "row is `absent`, and that is the answer.",
     "YOLO pose. Ни трансформера, ни нормализации, ни внимания — большая часть "
     "этой строки `absent`, и это и есть ответ."),
    ("hands", "Hand landmarks", "Ключевые точки кисти",
     "MediaPipe hands: a palm detector and a hand-landmark network in ONE "
     "container, run in that order with the whole decode between them. Seven of "
     "this row's eight cells are `absent` and one is `blocked`, and neither is "
     "an oversight.",
     "MediaPipe hands: детектор ладони и сеть ключевых точек в ОДНОМ контейнере, "
     "прогоняемые именно в этом порядке, со всем декодом между ними. Семь ячеек "
     "из восьми в этой строке `absent` и одна `blocked`, и ни то ни другое не "
     "недосмотр."),
    ("mppose", "Body pose", "Поза тела",
     "MediaPipe Pose: детектор человека и сеть позы в ОДНОМ контейнере, плюс "
     "одна операция, которой нет ни в одной другой архитектуре этого дерева. "
     "Семь ячеек `absent` и одна `blocked`, как и у рук, и по той же причине.",
     "MediaPipe Pose: детектор человека и сеть позы в ОДНОМ контейнере, плюс "
     "одна операция, которой нет ни в одной другой архитектуре этого дерева. "
     "Семь ячеек `absent` и одна `blocked`, как и у рук, и по той же причине."),
    ("embeddinggemma-300m", "Gemma", "Gemma",
     "The one model whose ENCODER differs from its kind, so it gets a row of its "
     "own rather than another kind.",
     "Единственная модель, чей КОДЕР отличается от своего `kind`, поэтому у неё "
     "своя строка, а не ещё один kind."),
]

# Status -> the one word that goes in the matrix. Short because a table cell has
# to stay a cell; the per-architecture sections carry the reasons.
MARK = {
    "honours": "**yes**",
    "on_array": "already",
    "unimplemented": "no code",
    "blocked": "blocked",
    "absent": "-",
}

MARK_RU = {
    "honours": "**да**",
    "on_array": "уже",
    "unimplemented": "нет кода",
    "blocked": "невозможно",
    "absent": "—",
}

STATUS_MEANING_RU = {
    "honours": "работает на массиве сегодня",
    "on_array": "работа уже диспатчится; коду нечего выбирать, и он "
                "отвергается с этим объяснением",
    "unimplemented": "операция у модели есть, ветка не написана",
    "blocked": "на этой плате перенести нельзя, по названной причине",
    "absent": "такой операции у модели нет",
}

# The eight codes' long names. Transcribed rather than translated: these are the
# names the runtime prints in its refusals (`--npu-ops softm (softmax)`), and a
# Russian document that renamed them would be quoting the runtime in a language it
# does not speak.
LONG_RU = {
    "gelu": "GELU",
    "layn": "LayerNorm",
    "softm": "softmax",
    "conv": "conv1d (аудио-фронтенд Whisper)",
    "attn": "внимание, как два GEMM",
    "mproj": "банк mel-фильтров Whisper, как GEMM",
    "fft": "трансформация 400 точек, как GEMM",
    "logit": "проекция в словарь, как GEMM",
}

# Why each of the five stream-only codes has no directory of its own. Translated
# from NO_DESIGN, same reason as the cell reasons below.
NO_DESIGN_RU = {
    "conv": "идёт на потоке, который `gemm_rtp` и так экспортирует, так что "
            "компилировать нечего",
    "attn": "его два потока добавляются в сами наборы `gemm_rtp` и "
            "`gemm_rtp_dec`, а не в отдельный каталог",
    "mproj": "его поток добавляется в сам набор `gemm_rtp`, а не в отдельный "
             "каталог",
    "fft": "его поток добавляется в сам набор `gemm_rtp`, а не в отдельный "
           "каталог",
    "logit": "его потоки добавляются в сам набор `gemm_rtp_dec`, а не в "
             "отдельный каталог",
}

# (architecture, code) -> the Russian reason. Every one of the 48 cells must be
# here; doc() raises on a missing one rather than falling back to English, because
# a Russian reader who hits one English cell among forty-eight has no way to tell
# that it is a bug rather than an oversight in the translation.
REASONS_RU = {
("gemm_rtp", "gelu"):
    "активация негейтед FFN — отдельный проход, поэтому его забирает дизайн "
    "`gelu/`. У ГЕЙТЕД FFN (nomic, gte, gemma) активация считается внутри "
    "гейтед-пути, и такого прохода нет — это и проверяет `buildable_codes()`. "
    "Та же ОГОВОРКА, что и в строке `cls`: точный erf получается только при "
    "`kind stt`, так что этот дизайн — фита `poly` 8-й степени, 2.49e-3 "
    "относительно от точного erf хоста. Замерено сквозным прогоном на bge-base с "
    "включённым кодом: relfro 6.1e-03 — поэтому рантайм принимает код, а не "
    "отказывает.",
("gemm_rtp", "layn"):
    "pre-LN: два LayerNorm на слой плюс финальный. Дизайн `layernorm/` "
    "строится под `d_model` и `layer_norm_eps` ЭТОЙ модели, и рантайм сверяет "
    "оба с контейнером, который держит, так что набор, экспортированный для "
    "другой модели, отвергается поимённо, а не нормализует молча неверными "
    "числами.",
("gemm_rtp", "softm"):
    "softmax над матрицей оценок — отдельный проход между двумя GEMM "
    "внимания, поэтому дизайн `softmax/` забирает его как есть. ЗАМЕРЕНО на "
    "bge-base, 15 текстов, arch 1: 1.70 с против 0.16 с хостовой эмбеддинг-"
    "функции, relfro 1.15e-02 / cos 0.999933 против хостовых векторов "
    "(max|d| 1.6e-03). Вся трудность тут — в ШИРИНЕ строки, и она была "
    "неверной, пока это не замерили: дизайн скомпилирован под sm_cols = "
    "паддированный n_kv (64 → 384, resolve.py:414), потому что NpuAttention, "
    "его второй потребитель, кладёт строки оценок в ширину самого ядра, а "
    "хостовый qk_impl клал строки шириной g_seq. `elt_chunks` этого не ловит — "
    "его единственная проверка `n % cols != 0`, а batch*heads*g_seq*g_seq и так "
    "делится на 384, — так что ядро нормализовало окна по 384 элемента, то "
    "есть шесть настоящих строк оценок за раз, и эмбеддинг выходил с relfro "
    "7.08e-01 / cos 0.749237895. Теперь BertEncoder раскладывает строки в "
    "ширину дизайна с `-1e30` за g_seq, как только массив берёт softmax; путь "
    "по умолчанию байт-в-байт совпадает с прежним, а `attn,softm` (relfro "
    "1.21e-02) не изменился. ЦЕНА — и это не дефект этой ячейки, а то, как "
    "комбинируются два флага: при attn,softm softmax оказывается внутри "
    "NpuAttention и диспатчится раз на голову и на чанк, а не раз на слой, и "
    "каждый диспатч заполняет всю row-capacity дизайна в 12288 строк — 100.90 с "
    "на 15 текстов против 1.38 с для одного attn.",
("gemm_rtp", "conv"):
    "называет conv1/conv2 Whisper — аудио-фронтенд, а этой архитектуры нет.",
("gemm_rtp", "attn"):
    "`qk()` и `av()` в `BertEncoder::run` диспатчатся на собственные потоки "
    "`attn_qk`/`attn_av` набора через общий NpuAttention. Слитый qkv BERT — это "
    "Q|K|V на позицию, поэтому блок K|V, которого хочет класс, достигается "
    "передачей qkv + d_model со сдвигом 3*d_model — тот же отступ, что "
    "использует собственное внимание Whisper, и никакого нового кода. Экспортёр "
    "строит потоки из реестра, как любой другой исполняемый код; n_kv берётся "
    "из `max_seq_len` ЦЕЛИ, то есть контекста, под который УПАКОВАН контейнер "
    "(all-MiniLM: 256, подрезано), а не 512 позиций чекпойнта — чтение "
    "config.json построило бы панель шириной 512 поверх тензора, который не "
    "адресует дальше 256. Паддинг за живыми позициями точен: NpuAttention "
    "пишет `-1e30` в эти колонки до softmax, так что больший n_kv стоит "
    "арифметики, а не точности. K у attn_qk — это head_dim, паддированный до "
    "tile_k, и именно это делает возможным голову шириной 32 (три из восьми "
    "моделей); рантайм зануляет соответствующие строки собранного блока Q, так "
    "что лишние MAC дают 0*0. Та же ОГОВОРКА, что в строках `cls` и `stt`: "
    "внимание на массиве МЕДЛЕННЕЕ хостового прохода. ЗАМЕРЕНО на bge-base, 15 "
    "текстов, arch 1: 1.38 с против 0.16 с хостовой эмбеддинг-функции (в 8.6 "
    "раза МЕДЛЕННЕЕ), relfro 1.58e-02 / cos 0.999876, max|d| 2.2e-03 — против "
    "4.6x у Whisper. Этот код делает модель запускаемой на массиве, а не "
    "быстрее.",
("gemm_rtp", "mproj"):
    "называет банк mel-фильтров Slaney — аудио-фронтенд, а этой архитектуры нет.",
("gemm_rtp", "fft"):
    "называет трансформацию 400 точек — аудио-фронтенд, а этой архитектуры нет.",
("gemm_rtp", "logit"):
    "называет СВЯЗАННЫЙ ЭМБЕДДИНГ Whisper, используемый как матрица логитов "
    "(`decoder.cpp:196`: в чекпойнте нет `proj_out`). Эмбеддер заканчивается "
    "головой пулинга и такого тензора не несёт: в `bert_encoder.cpp` ноль "
    "вхождений `logit` или `vocab`.",
("stt", "gelu"):
    "чекпойнт Whisper объявляет `activation: gelu`, а упаковщик отвергает всё "
    "остальное, так что это точное erf-ядро. `poly`-вариант на 2.49e-3 дальше от "
    "него — это уже другая активация, а не более быстрая та же.",
("stt", "layn"):
    "pre-LN с собственным `layer_norm_eps` этого чекпойнта (1e-05, внутри "
    "квадратного корня в `kernels/layernorm.cc`), строится под цель, а не по "
    "умолчанию.",
("stt", "softm"):
    "softmax над матрицей оценок — отдельный проход; ширина строки дизайна "
    "следует геометрии внимания, рядом с которым он построен.",
("stt", "conv"):
    "conv1/conv2 идут на СОБСТВЕННОМ потоке энкодерного набора `[rows, d, d]` — "
    "shape строки `attn_out`, — поэтому код не стоит ни одного `hw_context` и ни "
    "одного лишнего xclbin. Это единственная операция, выигрывающая по обеим "
    "осям: стоимость на хосте растёт как d², на массиве — как число диспатчей, "
    "а оно постоянно.",
("stt", "attn"):
    "два GEMM по краям softmax, как потоки `attn_qk` и `attn_av` в том же "
    "наборе. ЗАМЕРЕНО 4.32 с против 0.94 с на хосте на окне в 3 с: работает, и "
    "в 4.6 раза медленнее.",
("stt", "mproj"):
    "банк mel — ещё один поток в энкодерном наборе, при (201 бине, столько mel "
    "у модели).",
("stt", "fft"):
    "трансформация 400 точек фронтенда — ещё один поток в энкодерном наборе, "
    "при (400, 201).",
("stt", "logit"):
    "связанный эмбеддинг, транспонированный и нарезанный на восемь потоков-"
    "чанков в ДЕКОДЕРНОМ наборе; 39 МБ подложенных панелей. Нарезан потому, "
    "что 51865 колонок — это 1621 плитка по 32 против 12 у крупнейшего "
    "корабльного дизайна.",
("cls", "gelu"):
    "pre-LN с негейтед FFN, так что активация ЯВЛЯЕТСЯ отдельным проходом, и его "
    "забирает дизайн `gelu/`. ЗАМЕРЕНО на `vit-base-patch16-224`, bf16, "
    "bus.jpg, лучший из 8: 0.343 с против 0.248 с у хоста — в 1.38 раза "
    "МЕДЛЕННЕЕ. ОГОВОРКА, и она про точность, а не про код: `resolve.py` "
    "форсирует точное erf-ядро только для `kind stt`, а ViT остаётся на "
    "дефолте экспортёра `poly`, то есть это фита 8-й степени — 2.49e-3 "
    "относительно от точного erf, которое считает хост. Замеренный дрейф на "
    "наборе меток — меньше 0.011 по уверенности, поэтому это оговорка, а не "
    "отказ. Цели, которой нужно точное ядро, приходится сказать об этом в "
    "своих overrides; в дереве этого не делает никто.",
("cls", "layn"):
    "25 мест (два на слой плюс финальный), то же ядро. ЗАМЕРЕНО 0.377 с против "
    "0.248 с — в 1.52 раза медленнее.",
("cls", "softm"):
    "здесь softmax ВНУТРИ внимания, поэтому ячейка называет его массивной "
    "половиной того прохода, и он ЗАМЕРЕН сам по себе: vit-base-patch16-224, "
    "bus.jpg, arch 5, `--npu-ops softm` даёт энкодеру 0.822 с против 0.244 с "
    "у хоста (61 диспатч, из них 49 — GEMM и 12 — по одному softmax на блок), "
    "против 0.349 с для одного `attn`. Точность проверяется на ВСЕЙ строке "
    "вероятностей из 1000 компонент, а не на top-5, который совпал бы в любом "
    "случае: relfro 1.959e-02, cos 0.999869, max|d| 1.02e-02, метка и весь "
    "top-5 совпадают с хостом, а сдвиг центрированных логитов в top-10 меньше "
    "5.3e-02. Ширина дизайна — паддированное число ключей (197 → 384 против "
    "lcm(tile_k, tile_n)=192), потому что npue::whisper::attention — а это и "
    "ЕСТЬ хостовый путь этой модели — раскладывает строки оценок в ширину "
    "самого ядра, так что влезает целая строка на один диспатч, а не 64 её "
    "колонки. ЦЕНА запроса обоих кодов сразу, названа потому, что это "
    "единственный результат, которого читатель не угадает: `attn,softm` — "
    "6.685 с. Тогда "
    "NpuAttention зовёт softmax раз на голову и на чанк — 144 лишних диспатча — "
    "и каждый диспатч заполняет всю row-capacity дизайна в 12288 строк, "
    "каков бы ни был запрос вызывающего. Ячейка исполняет флаг; комбинация "
    "исполняет его медленно.",
("cls", "conv"):
    "patch embedding — это свёртка `Conv2d(3, d, kernel=16, stride=16)`, и она "
    "УЖЕ на массиве. im2col делает из неё `[n_patches, patch_dim] x "
    "[patch_dim, d]`, то есть ровно shape строки `attn_out`, поэтому "
    "диспатчится она в слот инструкции `attn_out` вообще без флага "
    "(`runtime/src/vit/encoder.cpp:267`). Запрос `conv` не добавил бы ни одного "
    "диспатча. Переписывание точное на одной только почве: stride равен kernel "
    "при padding 0, так что 196 окон ни не перекрываются, ни пропускаются, и "
    "im2col — это перестановка, а не сумма.",
("cls", "attn"):
    "12 голов на 197 позициях, на собственных потоках `attn_qk`/`attn_av` "
    "набора — экспортёр строит их из этой записи реестра, как любой другой "
    "исполняемый код, а n_kv — это `max_seq_len` контейнера (197), "
    "паддированный до 384 против lcm(tile_k, tile_n). У ViT своего внимания "
    "нет: `vit/encoder.cpp` зовёт `npue::whisper::attention()`, то есть "
    "добавленная тут ветка на массив — это ровно тот NpuAttention, который уже "
    "есть, достижимый через тот же общий хостовый путь. ЗАМЕРЕНО на "
    "vit-base-patch16-224, bus.jpg, arch 5: 0.349 с против 0.244 с на весь "
    "энкодер (в 1.43 раза МЕДЛЕННЕЕ), 337 диспатчей против 49. Точность на "
    "всей 1000-компонентной строке вероятностей: relfro 4.384e-03, cos "
    "0.999993, max|d| 2.1e-03, метка и top-5 совпадают с хостом, сдвиг "
    "центрированных логитов в top-10 меньше 1.9e-02 — та самая строка в 197 "
    "позиций, так что ничто тут не опирается на seq 64, на котором "
    "останавливались старые заметки. Цена выше 64 позиций — теперь это число, "
    "замеренное, а не оставленное открытым.",
("cls", "mproj"):
    "банк mel-фильтров — часть аудио-фронтенда, а этой архитектуры нет.",
("cls", "fft"):
    "трансформация 400 точек — часть аудио-фронтенда, а этой архитектуры нет.",
("cls", "logit"):
    "классификационная голова — настоящая проекция `[768, 1000]`, так что это не "
    "отсутствующая операция, — но 1000 не кратно `tile_n*cols = 48*4 = 192`, "
    "поэтому на этом массиве не существует ни одной законной панели B такой "
    "ширины, и построить её нельзя. Это не «не хватает»: на этой плате её не "
    "существует. (Впрочем, это 0.8% стоимости картинки в виде хостового "
    "matvec, так что по скорости вопрос пустой.)",
("pose", "conv"):
    "72 диспатченные свёртки, на собственных потоках `convNNxMM` набора позы "
    "(у шима 16 дескрипторов DMA-буферов — поэтому форм 14, а не одна). ЗАМЕРЕНО "
    "290 мс против 150 мс у хоста на 640×640: здесь в 1.4 раза МЕДЛЕННЕЕ, и "
    "сообщение рантайма об отказе для контейнера без набора дизайнов говорит "
    "ровно это.",
("pose", "gelu"):
    "активация здесь SiLU, а не GELU, и она ВФЬЮЖЕНА в эпилог свёртки: "
    "`Mul(x, Sigmoid(x))` — графовая операция, которую упаковщик узнаёт как "
    "паттерн активации. Отдельного прохода, который можно перенести, нет.",
("pose", "layn"):
    "операции нормализации в этой сети нет вообще. `GRAPH_OPS` в "
    "`packers/pose.py` — это `{Conv, Mul, Sigmoid, Add, Concat, Split, MaxPool, "
    "Resize}`, и BatchNormalization там отсутствует, так что граф с BN был бы "
    "ОТВЕРГНУТ ПОИМЕННО — ultralytics сворачивает BN в веса свёрток на экспорте, "
    "вот почему `grep` не находит BatchNormalization нигде в `tools/`.",
("pose", "softm"):
    "нет внимания, значит нет softmax.",
("pose", "attn"):
    "нет внимания: `net.hpp` открывает conv/concat/slice/add/maxpool/upsample/"
    "head и больше ничего взвешенного.",
("pose", "mproj"):
    "банк mel-фильтров — часть аудио-фронтенда, а этой архитектуры нет.",
("pose", "fft"):
    "трансформация 400 точек — часть аудио-фронтенда, а этой архитектуры нет.",
("pose", "logit"):
    "словаря нет: голова — это свёртка 1×1, дающая один class score и сетку DFL, а "
    "не проекция в пространство токенов.",
("hands", "conv"):
    "Оно ДИСПАТЧИТ, и здесь в 1.8 раза МЕДЛЕННЕЕ. Оба факта измерены, и оба "
    "должны быть в одной ячейке.\n\n"
    "Что изменилось и когда: эта ячейка была `blocked` с причиной, что нет "
    "набора дизайнов под плотные пары (K, N) arch=7. Это было верно, когда "
    "писалось, и перестало быть верно, когда в "
    "`tools/export/exporters/gemm_rtp/geometry.py` появился `HANDS_CONV_SHAPES` "
    "и из него собрали "
    "`runtime/artifacts/mediapipe-hands/artifacts_npu1/gemm_rtp/`: двенадцать "
    "потоков, один xclbin, сверенный с упакованным контейнером в обе стороны "
    "через `tools/verify/verify_pose_streamset.py`. `--npu-ops conv` отправляет "
    "туда 61 плотную свёртку за 149 диспатчей; остальные 39 остаются на хосте.\n\n"
    "ИЗМЕРЕНО на hand_plain.png, одна рука, по пять прогонов: массив берёт "
    "155 мс против хостовских 85 мс. Из времени массива 47 мс — само устройство "
    "(GEMM), 4.5 мс — перепаковка A в паддинговый шаг дизайна, 3.6 мс — "
    "транспонирование C. То есть здесь УСТРОЙСТВО — большая часть, а хостовая "
    "перекладка мала, и это самый ясный в файле случай, когда массив проигрывает "
    "на устройстве, а не на перемещении данных вокруг него. N паддится до "
    "tile_n*cols = 128, а числа каналов здесь 16, 24, 32, 48, 64, 96, 128, 192 "
    "и 256, так что большинство паддится, и полезные 356M плотных MAC "
    "превращаются в заметно больший объём диспатчимой работы.\n\n"
    "И ДВА ПУТИ РАСХОДЯТСЯ ГДЕ-ТО НА ПОЛТОРА ПИКСЕЛЯ. Панели в bf16, поэтому "
    "score детектора сдвигается на 1.1e-02, его бокс — на 1.08 px, а 21 точка "
    "сети ключевых точек попадает в среднем на 1.63 px от хостовских и хуже всего "
    "на 4.71 px (медиана 1.72, p90 2.45); глубина — в среднем 0.41 px и хуже "
    "всего 2.17 px, presence в пределах 3.4e-04, мировые точки — 0.0023 m. Это "
    "тот же порядок, что у массивного пути arch=6, который сходится с хостом до "
    "1.1 px, и ЛУЧШЕ, чем у arch=8, где шум детектора усиливается повёрнутым "
    "кропом до 44 px: кроп arch=7 не поворачивается на угол, предсказанный "
    "детектором, так что здесь ничто не умножает ошибку.\n\n"
    "39 из 100 свёрток контейнера — ГЛУБИННЫЕ, несут 77.9M MAC, а фильтр "
    "depthwise сводит внутри одного канала, то есть `[M, N]`-GEMM в нём нет: они "
    "остаются на хосте, что бы набор ни оказался.",
("hands", "gelu"):
    "активации здесь ReLU, ReLU6 и PReLU, и каждая ВФЬЮЖЕНА в эпилог своей "
    "свёртки: `packers/hands.py` вешает `act` на Conv, dwconv или Add, и "
    "отдельного прохода нет ни у одной из трёх. Дизайну `gelu/` нечего "
    "забирать.",
("hands", "layn"):
    "операции нормализации нет ни в одном из двух графов. Инвентарь операций в "
    "`packers/hands.py` — `{Conv, Add, MaxPool, Pad, Resize}` для детектора "
    "ладони и `{Conv, Add, MaxPool}` для сети ключевых точек, и ни "
    "BatchNormalization, ни InstanceNormalization там нет, что было бы отвергнуто "
    "ПОИМЕННО, — чекпойнты MediaPipe не несут ни одного, и потому depthwise-слои "
    "это просто свёртки, а не separable-нормализация.",
("hands", "softm"):
    "нет ни внимания, ни softmax. Ближайшее к этому — score детектора ладони, и "
    "это СИГМОИДА, свёрнутая в декод (`runtime/src/hands/decode.cpp` применяет "
    "её к логиту после графа), а не softmax над матрицей оценок между двумя "
    "GEMM.",
("hands", "attn"):
    "нет внимания ни в одном графе: argmax по 63 каналам тепловой карты у сети "
    "ключевых точек свёрнут в граф как reshape, а `runtime/src/hands/net.cpp` "
    "обходит conv, add, maxpool, pad и resize и больше ничего взвешенного.",
("hands", "mproj"):
    "банк mel-фильтров — часть аудио-фронтенда, а этой архитектуры нет.",
("hands", "fft"):
    "трансформация 400 точек — часть аудио-фронтенда, а этой архитектуры нет.",
("hands", "logit"):
    "словаря нет: голова ключевых точек выдаёт 21 экранную точку, presence и "
    "handedness, а голова ладони — боксы и score. Нигде в этой архитектуре "
    "проекции в пространство токенов.",
("mppose", "conv"):
    "Оно ДИСПАТЧИТ, и здесь оно в 2.4 раза МЕДЛЕННЕЕ, и оно НЕ СОВПАДАЕТ с "
    "хостовским путём в пределах пикселя. Все три вещи измерены, и все три "
    "должны быть в одной ячейке.\n\n"
    "У 99 плотных свёрток двух графов 53 РАЗНЫЕ «сырые» пары (K, N), которые "
    "паддятся, как паддит позу `gemm_rtp/geometry.py` — K до кратного "
    "`tile_k = 64`, N до `tile_n*cols = 128` — до ДВАДЦАТИ ДВА. Это самый "
    "большой набор стримов в этом файле, и "
    "`runtime/artifacts/mediapipe-pose/artifacts_npu1/gemm_rtp/` теперь его "
    "носит: один дизайн, 22 потока, M = 1024, и "
    "`tools/verify/verify_pose_streamset.py` сверяет его с geometry.py, "
    "npu_targets.json и упакованным контейнером в обе стороны. `--npu-ops conv` "
    "отправляет эти 99 свёрток на массив за 447 диспатчей; остальные 62 остаются "
    "на хосте.\n\n"
    "ИЗМЕРЕНО на docs/bus.jpg, один человек: массив берёт 327 мс против "
    "хостовских 137 мс на том же кадре. Из этих 327 мс самого устройства 134 мс "
    "(GEMM), 56 мс — перепаковка A в паддинговый шаг дизайна, 24 мс — "
    "транспонирование C обратно. То есть устройство это 41 %, а хостовая "
    "перекладка вокруг диспатча — 80 мс, и честная причина, по которой массив "
    "здесь проигрывает, это не медленное устройство.\n\n"
    "И ДВА ПУТИ НЕ СОВПАДАЮТ. Панели массива в bf16, поэтому детектор отличается "
    "от хостовского на 7.7e-04 по score, 0.48 px по боксу и 1.37 px по "
    "ключевым точкам — обычный bf16-шум, и у arch=6 массив сходится с хостом до "
    "1.1 px. Здесь он УСИЛЕН, и геометрией, а не сетью: кроп — это квадрат, "
    "повёрнутый на угол, который задают ДВЕ ключевые точки ДЕТЕКТОРА, так что "
    "шум 1.37 px — это другой поворот кропа в 565 px, и точки, которые при "
    "пересемплировании страдают больше всего, выходят на 44 px от хостовских при "
    "медиане 5.1 px и среднем 8.4 px. Уверенность позы едет вместе с ними: "
    "0.9422 -> 0.9832.\n\n"
    "Это усиление НЕ означает, что массивная сеть неправа, и как это "
    "установлено — стоит сказать, потому что это единственное доказательство в "
    "этой ячейке, которое их разделяет: зоопарк OpenCV, независимая "
    "реализация, получил ДЕТЕКЦИЮ массивного пути и ответил conf 0.9809 с "
    "bbox [148.2, 333.3, 392.7, 885.2] против 0.9832 и "
    "[144.1, 334.1, 401.5, 890.2] у массивного рантайма, и 0.9449 с "
    "ХОСТОВСКОЙ детекцией против 0.9422 у хостового рантайма. То есть bf16-шум "
    "детектора объясняет всё расхождение целиком, и кто сравнивает два пути на "
    "этой архитектуре, сравнивает два кропа, а не две сети.\n\n"
    "ДВЕ ВЕЩИ, КОТОРЫЕ НАБОР ДИЗАЙНОВ НЕ СДВИНЕТ. 62 из 161 свёрток "
    "контейнера — ГЛУБИННЫЕ и несут 77.2M MAC, то есть 12.3 %, а фильтр "
    "depthwise сводит внутри одного канала, то есть `[M, N]`-GEMM в нём нет; "
    "они остаются на хосте, что бы набор ни оказался. И у этой архитектуры есть "
    "операция, которая вообще не GEMM и никогда им не станет: три шага "
    "DepthToSpace, чистые копии плоскостей, которые строят пирамиду детектора "
    "28/14/7 из одной карты 7×7.",
("mppose", "gelu"):
    "активации здесь ReLU и ReLU6, и каждая ВФЬЮЖЕНА в эпилог своей свёртки "
    "или в остаточный add: `packers/mppose.py` вешает `act` на Conv, dwconv или "
    "Add, и отдельного прохода нет ни для одной из двух. Дизайну `gelu/` нечего "
    "забирать.",
("mppose", "layn"):
    "операции нормализации нет ни в одном из двух графов. Инвентарь операций — "
    "`{Conv, DwConv, Add, MaxPool, Resize, DepthToSpace}` для детектора (три его "
    "пространственных Pad сворачиваются в шесть потребляющих их Conv, поэтому "
    "отдельными узлами они не идут) и `{Conv, DwConv, Add, MaxPool, Resize}` для "
    "сети позы. Ни BatchNormalization, ни InstanceNormalization там нет, что было "
    "бы отвергнуто ПОИМЕННО, — чекпойнты MediaPipe не несут ни одного, и потому "
    "depthwise-слои это просто свёртки, а не separable-нормализация.",
("mppose", "softm"):
    "нет ни внимания, ни softmax. Ближайшее к этому — score детектора, и это "
    "СИГМОИДА, которую несёт САМ ГРАФ, — в отличие от рук, где сигмоида головы "
    "ладонной свёрнута в декод и применяется к логиту в "
    "`runtime/src/hands/decode.cpp`, — потому что здесь упаковщик записывает "
    "`sigmoid` у выхода уверенности сети позы, и рантайм применяет её там. Ни то "
    "ни другое не softmax над матрицей оценок между двумя GEMM.",
("mppose", "attn"):
    "нет внимания ни в одном графе: пять выходов сети позы — это пять "
    "ОТДЕЛЬНЫХ свёрток, одна из них транспонируется на выходе, а одна "
    "сжимается сигмоидой, и `runtime/src/mppose/net.cpp` обходит conv, dwconv, "
    "add, maxpool, resize и d2s и больше ничего взвешенного.",
("mppose", "mproj"):
    "банк mel-фильтров — часть аудио-фронтенда, а этой архитектуры нет.",
("mppose", "fft"):
    "трансформация 400 точек — часть аудио-фронтенда, а этой архитектуры нет.",
("mppose", "logit"):
    "словаря нет: голова ключевых точек выдаёт 39 строк "
    "x/y/z/visibility/presence, уверенность, маску 256×256 и тепловую карту "
    "64×64×39, а голова детектора — боксы, четыре точки и score. Нигде в этой "
    "архитектуре проекции в пространство токенов.",
("embeddinggemma-300m", "gelu"):
    "GeGLU считает активацию ВНУТРИ гейтед-пути, между `ffn_up` и `ffn_down`, "
    "так что отдельного прохода, который мог бы забрать `hw_context`, нет, и "
    "исполнение кода напечатало бы «на МАССИВЕ», не сдвинув ничего. Измерено, а "
    "не заявлено: рантайм однажды взял `host_gelu` прямо из флага, напечатал "
    "строку МАССИВА для дизайна, который так и не открыл, и вернул векторы с "
    "relfro 0.000e+00 против собственного хостового пути (6.1e-03 у негейтед "
    "bge-base, тот же флаг).",
("embeddinggemma-300m", "layn"):
    "gemma нормализует RMSNorm, а не LayerNorm: нет прохода по среднему, нет "
    "beta, свой `rms_norm_eps`. `kernels/layernorm.cc` параметризован только "
    "`-DLN_COLS/-DLN_EPS/-DLN_ROWS`, так что это другое тело ядра, новый вид "
    "дизайна и ДЕВЯТЫЙ код. Намеренно не сделано: набор кодов стал бы больше и "
    "менее однородным — три из четырёх трансформерных архитектур носили бы код "
    "нормализации, а одна нет. Отвергнуто, а не сделано двумя разными "
    "операциями под одним именем.",
("embeddinggemma-300m", "softm"):
    "`GemmaNpuEncoder::attention` читает тот же пооперационный флаг, что и "
    "строка `gemm_rtp`: набор диспатчей энкодера строится из этой записи "
    "реестра, так что запрос `softm` доходит до него, как и запрос `attn`. "
    "Softmax тут ВНУТРИ внимания, поэтому это массивная половина того одного "
    "прохода: широкий дизайн softmax идёт по чанку оценок, который NpuAttention "
    "уже подготовил, при n_kv = 512 (скользящее окно, паддировано до 576 "
    "против lcm(tile_k, tile_n)=192), а не 2048 контейнера. ЗАМЕРЕНО на "
    "embeddinggemma-300m, arch 1: колонка внимания в разбивке идёт с 21 мс "
    "хостового пути на 1497 мс с `softm`, то есть 96 диспатчей становятся 96 + "
    "24 softmax, при relfro 6.227e-03 против хостовых векторов. Запрос "
    "`attn,softm` сразу — это выброс, и он назван измеренным: 18711 мс и 672 + "
    "288 диспатчей, relfro 6.664e-03. Причина в одну строку — NpuAttention "
    "зовёт softmax раз на голову и на чанк, пока каждый вызов заполняет всю "
    "row-capacity дизайна в 12288 строк, — и именно поэтому этот флаг стоит "
    "хотеть ради точности на массиве, а не ради скорости.",
("embeddinggemma-300m", "attn"):
    "`attention()` диспатчится на собственные потоки `attn_qk`/`attn_av` "
    "набора энкодера, но ДОЙТИ до Whisper-овского NpuAttention сменой сдвига "
    "она не может, и причину стоит записать, потому что это единственное место, "
    "где геометрия этой модели отличается по-настоящему, а не другими числами. "
    "Тот класс берёт ОДИН `d_model` и использует его трижды: отступ от строки "
    "ключа к её значению (d_model), сдвиг строки выхода (d_model) и число "
    "голов (d_model / head_dim). Для BERT и Whisper все три числа одинаковы — "
    "K и V по ширине ровно по одному d_model. Gemma — MULTI-QUERY: 3 головы "
    "запросов по 256 над ОДНОЙ ключ-значением головой в 256, то есть "
    "половинная ширина K|V равна 256, строка выхода — 768, а число голов — 3. "
    "Один параметр не может нести все три, поэтому класс получает отдельный "
    "kv_width, и именно эта модель его требует. Второе, что принадлежит только "
    "ей: её внимание ПОЛОСАТОЕ — sliding_window 512, каждый шестой слой, — "
    "когда контейнер упакован под 2048. NpuAttention маскирует СУФФИКС строки "
    "оценок, то есть паддинг за последовательностью, а полоса — не суффикс, "
    "поэтому n_kv равно окну, а рантайм ОТВЕРГАЕТ последовательность длиннее "
    "него, вместо того чтобы считать полное внимание там, где модель считает "
    "локальное. RoPE применяется к буферу qkv до вызова прохода, так что "
    "операнд A на массиве — активации уже после RoPE, и переносить ничего не "
    "нужно. ЗАМЕРЕНО здесь, а не позаимствовано у Whisper: на "
    "embeddinggemma-300m, arch 1, колонка внимания в разбивке идёт 21 мс "
    "(хост) → 358 мс (`attn`), 96 диспатчей становятся 672, при relfro "
    "5.735e-03 против хостовых векторов — в 17 раз медленнее на этой модели, "
    "тогда как собственное attn Whisper в 4.6 раза. Медленнее на обоих, так что "
    "этот код делает модель запускаемой на массиве, а не быстрее.",
("embeddinggemma-300m", "conv"):
    "называет conv1/conv2 Whisper — аудио-фронтенд, а этой архитектуры нет.",
("embeddinggemma-300m", "mproj"):
    "называет банк mel-фильтров Whisper — аудио-фронтенд, а этой архитектуры нет.",
("embeddinggemma-300m", "fft"):
    "называет трансформацию 400 точек Whisper — аудио-фронтенд, а этой "
    "архитектуры нет.",
("embeddinggemma-300m", "logit"):
    "называет связанный эмбеддинг, используемый как матрица логитов; в "
    "`gemma_npu_encoder.cpp` ноль вхождений `logit` или `vocab`.",
}


def reg_of(label):
    """The row for one architecture, whether it is a kind or a lone model.

    MODEL_REGISTRY wins over KIND_REGISTRY because a model entry REPLACES its
    kind's row rather than patching it -- registry_for() does the same thing for
    the exporter, and two places answering that question differently is the bug
    this function exists to prevent.
    """
    if label in npu_ops.MODEL_REGISTRY:
        return npu_ops.registry_for(None, label)
    return npu_ops.KIND_REGISTRY[label]


def cell_rows(label):
    """(code, long name, status, reason) in CODE order, for one architecture."""
    reg = reg_of(label)
    return [(code, npu_ops.OPS[code][1], reg[code][0], reg[code][1])
            for code in npu_ops.OPS]


def ru_reason(label, code):
    """The Russian reason, or a loud failure if the translation is missing.

    Not a fallback to English. One English cell among forty Russian ones reads as
    an oversight in the translation rather than as a hole, so the hole has to be
    the loud kind: the generator refuses to produce the file, and the matrix gate
    fails.
    """
    try:
        return REASONS_RU[(label, code)]
    except KeyError:
        raise SystemExit(
            f"NPU_OPS.ru.md: no Russian reason for {label}/{code}. The English "
            f"registry has one (tools/lib/npu_ops.py, KIND_REGISTRY or "
            f"MODEL_REGISTRY), so the translation is behind the source. Add it to "
            f"REASONS_RU in {Path(__file__).name} -- do not leave the cell "
            f"English."
        ) from None


def matrix_table(ru=False):
    """One row per architecture, listing the codes it can move -- NOT a grid.

    Same shape as NPU_MODELS.md's table and for the same reason, which is the
    reason it is written here too rather than left as the one grid in the tree: of
    56 cells, 31 said the architecture has no such operation, so more than half
    the table was one absence repeated, and the answer to "what can this dispatch"
    came from filtering the empty cells out of a row.

    The cell is the status and nothing else. An earlier version put the op's long
    name under it and the table was unreadable -- worse, the names are WRONG for
    some cells: `conv` is Whisper's audio front end in the gemm_rtp and stt rows
    and a ViT's patch embedding in the cls row, and one string per column cannot
    say both. The names are in the per-architecture section below, where the
    reason for each cell has room to explain what that cell's op actually is.
    """
    # One column, every entry naming its own status -- NPU_MODELS.md's shape and
    # for NPU_MODELS.md's reason, because a half-answered table is worse than
    # either answer: 31 of 56 cells saying the same absence here while that file
    # stopped saying it would leave the reader with two tables and one rule.
    suffix = {npu_ops.BLOCKED: "невозможно" if ru else "impossible",
              npu_ops.ON_ARRAY: "уже" if ru else "already",
              npu_ops.UNIMPLEMENTED: ("нет кода" if ru else "no code")}
    # "дополнительно" for the reason in gen_npu_models_doc.py's copy of this
    # header: the eight codes name what a caller can move ON TOP OF an
    # architecture's own stream set. For a transformer that set is four GEMM
    # streams which dispatch by default and which no code names, so a row listing
    # four extras and no `conv` is not a row that never touches the array -- it is
    # a row that touches it twelve times before anybody types a flag. For `pose`
    # and `mppose` the convolutions ARE that set, which is why their `conv` reads
    # `есть` and their rows are one code long.
    head = ("| архитектура | что можно дополнительно отправить на массив |" if ru
            else "| architecture | what can additionally go to the array |")
    lines = [head, "| --- | --- |"]
    for entry in ARCHES:
        label = entry[0]
        reg = reg_of(label)
        items = []
        for c in npu_ops.OPS:
            st = reg[c][0]
            if st == npu_ops.HONOURS:
                items.append(f"`{c}`")
            elif st in suffix:
                items.append(f"`{c}` ({suffix[st]})")
        lines.append(f"| `{label}` | " + (", ".join(items) or "—") + " |")
    return lines


def _are(n):
    """'is' for 1, 'are' otherwise.

    The tally is generated and the sentence around it is not, so the two drift
    apart the moment a cell changes status -- "1 are already there" is exactly
    what a table with one `on_array` cell produces, and it is the sort of thing
    that makes a reader stop believing the numbers.
    """
    return "is" if n == 1 else "are"


def _do(n):
    return "does" if n == 1 else "do"


def doc(ru=False):
    counts = {s: 0 for s in npu_ops.STATUSES}
    for entry in ARCHES:
        reg = reg_of(entry[0])
        for code in npu_ops.OPS:
            counts[reg[code][0]] += 1
    n_codes, n_arch = len(npu_ops.OPS), len(ARCHES)

    L = []
    a = L.append
    if ru:
        a("# Какая архитектура какой код `--npu-ops` исполняет")
        a("")
        a("<!-- СГЕНЕРИРОВАНО tools/gen_npu_ops_doc.py из tools/lib/npu_ops.py.")
        a("     Не править руками: запустите скрипт, иначе упадёт")
        a("     tools/verify/verify_npu_op_matrix.py. Реестр — источник истины,")
        a("     этот файл — его проза. Английская версия: NPU_OPS.md -->")
        a("")
        a("У рантайма ОДИН флаг на это, `--npu-ops CODES`, и его значение по")
        a("умолчанию — пустое множество: **всякая операция считается на CPU, а")
        a("это измеренно более быстрый путь во всех случаях, где мерили.** Коды в")
        a("списке — это способ попросить массив вместо этого, по одной операции.")
        a("")
        a("У экспортёра такого флага НЕТ вовсе. Одна команда собирает каждый")
        a("дизайн, который цель способна использовать:")
        a("")
        a("```")
        a("python tools/export/export_gemm_rtp.py --target <model> --arch 1")
        a("```")
        a("")
        a("Она печатает выбранный список и причину для каждого пропущенного кода,")
        a("и не принимает никакого флага операций: `--npu-ops`, `--npu-extra-ops`")
        a("и `--npu-eltwise` там отвергаются поимённо, у каждого своё сообщение.")
        a("Ниже — то, что он читает.")
        a("")
        # The tally is a LIST OF STATUSES rather than the English version's prose
        # sentence, and that is not a stylistic choice: Russian agreement makes
        # "14 работают / 1 работает" a thing the template cannot get right for
        # every count, and a bullet carrying the status name is invariant under
        # any number. The English keeps its sentence because English is invariant
        # here and the sentence reads better than a list.
        a(f"Кодов {n_codes}, архитектур {n_arch}, ячеек {n_codes * n_arch}. "
          f"Из них:")
        a("")
        a(f"- **{counts['honours']}** `honours` — работает на массиве сегодня;")
        a(f"- **{counts['on_array']}** `on_array` — работа уже диспатчена без кода;")
        a(f"- **{counts['unimplemented']}** `unimplemented` — операция у модели "
          f"есть, ветка на массив не написана;")
        a(f"- **{counts['blocked']}** `blocked` — на эту плату перенести нельзя, "
          f"причина названа ниже;")
        a(f"- **{counts['absent']}** `absent` — такой операции у модели нет.")
        a("")
        a("## Восемь кодов")
        a("")
        a("Три — поэлементные дизайны собственной сборки, в соседнем каталоге")
        a("рядом с набором GEMM:")
        a("")
        for code in ("gelu", "layn", "softm"):
            a(f"- `{code}` — {LONG_RU[code]}, в `{npu_ops.OPS[code][0]}/`.")
        a("")
        a("Пять — НЕ дизайны собственной сборки. Это потоки, добавляемые в набор")
        a("дизайнов, который экспорт GEMM и так производит, так что запрос кода не")
        a("стоит лишнего xclbin — поэтому `build.py` для них ничего не")
        a("компилирует, и знать о них должен только `resolve.py`:")
        a("")
        for code in sorted(npu_ops.STREAM_ONLY):
            a(f"- `{code}` — {LONG_RU[code]}: {NO_DESIGN_RU[code]}.")
        a("")
        a("## Что значат пять статусов")
        a("")
        for s in npu_ops.STATUSES:
            a(f"- **{s}** — {STATUS_MEANING_RU[s]}.")
        a("")
        a("`unimplemented` стоит отличать от `blocked` внимательнее всего: это")
        a("**ветка кода, которую никто не написал**, а не свойство модели и не")
        a("то, что экспортёр мог бы починить сам. `blocked` значит, что")
        a("операция настоящая и здесь не переносится, и причина говорит какая из")
        a("трёх: вфьюжена в другую операцию, требует другого ядра и нового кода,")
        a("или не тайлится.")
        a("")
        a("## Что каждая архитектура может отправить на массив")
        a("")
        a("Список на архитектуру, а не сетка архитектура на коды: из 56 ячеек "
          f"{counts[npu_ops.ABSENT]} повторяли бы одно и то же отсутствие, и "
          "ответ на вопрос, что это может диспатчить, получался выбрасыванием "
          "пустых ячеек из строки. Здесь пустых ячеек нет: один столбец, и "
          "каждый его пункт называет свой статус сам — `conv` (уже) читается "
          "без заголовка в трёх колонках слева. Кода, которого в списке нет, у "
          "архитектуры нет.")
        a("")
        L.extend(matrix_table(ru=True))
        a("")
        a("`уже` — ячейка пуста потому, что работа уже сделана без кода, и это")
        a("лучше галочки. `—` — у модели нет такой операции, и это навсегда и")
        a("правильно. `нет кода` — честная середина: операция есть, и ничего до")
        a("неё не доходит.")
        a("")
        for label, title_en, title_ru, blurb_en, blurb_ru in ARCHES:
            a(f"## `{label}` — {title_ru}")
            a("")
            a(blurb_ru)
            a("")
            for code, long_name, status, _reason_en in cell_rows(label):
                a(f"### `{code}` — {LONG_RU[code]}: **{status}**")
                a("")
                a(ru_reason(label, code))
                a("")
                design = npu_ops.OPS[code][0]
                if not design and status == "honours":
                    a(f"Каталог дизайна: нет — {NO_DESIGN_RU[code]}. См. "
                      f"`STREAM_ONLY` в `tools/lib/npu_ops.py`.")
                    a("")
                elif status == "honours":
                    a(f"Каталог дизайна: `{design}/`, собирается экспортёром "
                      f"автоматически для любой цели этой архитектуры.")
                    a("")
        a("## Где это проверяется")
        a("")
        a("- `tools/lib/npu_ops.py` — реестр. Экспортёр читает его, решая что "
          "строить.")
        a("- `runtime/include/common/npu_ops_flag.hpp` — вторая копия восьми "
          "кодов и их длинных имён у рантайма. Ни одна сторона не может включить "
          "другую, поэтому каждая указывает на другую, а цена этого "
          "дублирования — отказ поимённо с обеих сторон, а не неверное число.")
        a("- `tools/verify/verify_npu_op_matrix.py` — проверяет, что этот файл "
          "ровно то, что печатает `tools/gen_npu_ops_doc.py`, что ячеек ровно "
          f"{n_codes * n_arch} и разложение по статусам верно, и что рантайм "
          "принимает или отвергает коды так, как здесь написано.")
        a("")
        return "\n".join(L)

    a("# Which architecture honours which `--npu-ops` code")
    a("")
    a("<!-- GENERATED by tools/gen_npu_ops_doc.py from tools/lib/npu_ops.py.")
    a("     Do not edit: run the script, or tools/verify/verify_npu_op_matrix.py")
    a("     fails. The registry is the source of truth; this file is its prose.")
    a("     Russian version: NPU_OPS.ru.md -->")
    a("")
    a("The runtime has ONE flag for this, `--npu-ops CODES`, and its default is")
    a("the empty set: **every op runs on the CPU, which is the measured-faster")
    a("path in every case measured so far.** Passing codes is how you ask for the")
    a("array instead, one op at a time.")
    a("")
    a("The exporter has **no such flag**. One command builds every design the")
    a("target can use:")
    a("")
    a("```")
    a("python tools/export/export_gemm_rtp.py --target <model> --arch 1")
    a("```")
    a("")
    a("It prints the list it chose and the reason for every code it skipped, and")
    a("it takes no op flag at all -- `--npu-ops`, `--npu-extra-ops` and")
    a("`--npu-eltwise` are refused by name there, each with its own message. The")
    a("table below is what it reads.")
    a("")
    a(f"There are {n_codes} codes and {n_arch} architectures: "
      f"{n_codes * n_arch} cells. Of those, "
      f"{counts['honours']} run on the array today, "
      f"{counts['on_array']} {_are(counts['on_array'])} already there without a "
      f"code, {counts['unimplemented']} {_are(counts['unimplemented'])} "
      f"operations the model has and no array branch reaches, "
      f"{counts['blocked']} cannot be moved on this board for a stated "
      f"reason, and {counts['absent']} "
      f"{_do(counts['absent'])} not exist in that model at all.")
    a("")
    a("## The eight codes")
    a("")
    a("Three are elementwise designs of their own, compiled into a sibling")
    a("directory next to the GEMM set:")
    a("")
    for code in ("gelu", "layn", "softm"):
        a(f"- `{code}` -- {npu_ops.OPS[code][1]}, in `{npu_ops.OPS[code][0]}/`.")
    a("")
    a("Five are **not** designs of their own. They are streams added to a design")
    a("set the GEMM export already produces, so asking for them costs no extra")
    a("xclbin -- which is why `build.py` compiles nothing for them and only")
    a("`resolve.py` has to know they exist:")
    a("")
    for code in sorted(npu_ops.STREAM_ONLY):
        a(f"- `{code}` -- {npu_ops.OPS[code][1]}: "
          f"{npu_ops.NO_DESIGN[code]}.")
    a("")
    a("## What the five statuses mean")
    a("")
    for s in npu_ops.STATUSES:
        a(f"- **{s}** -- {npu_ops.STATUS_MEANING[s]}.")
    a("")
    a("`unimplemented` is the one worth distinguishing carefully from `blocked`:")
    a("it is **a branch of code that was never written**, not a property of the")
    a("model and not something the exporter can fix on its own. `blocked` means")
    a("the operation is real and cannot be moved here, and the reason says which")
    a("of the three it is: fused into another op, needs a different kernel and a")
    a("new code, or does not tile.")
    a("")
    a("## What each architecture can send to the array")
    a("")
    a("A list per architecture, not a grid of architectures against codes: of 56 "
      f"cells {counts[npu_ops.ABSENT]} repeated one absence, and the answer to "
      "what can this dispatch came from filtering the empty cells out of a row. "
      "There are no empty cells here: one column, and every entry in it names its "
      "own status -- `conv` (already) reads without a heading three columns to "
      "the left. A code that is not listed is not one this architecture has.")
    a("")
    L.extend(matrix_table())
    a("")
    a("`already` means the cell is empty because the work is done without a code,")
    a("which is better than a tick. `-` means the model has no such operation,")
    a("which is permanent and correct. `no code` is the honest middle: the")
    a("operation is there, and nothing reaches it.")
    a("")
    for label, title_en, title_ru, blurb_en, blurb_ru in ARCHES:
        a(f"## `{label}` -- {title_en}")
        a("")
        a(blurb_en)
        a("")
        for code, long_name, status, reason in cell_rows(label):
            a(f"### `{code}` -- {long_name}: **{status}**")
            a("")
            a(reason)
            a("")
            design = npu_ops.OPS[code][0]
            if not design and status == "honours":
                a(f"Design directory: none -- "
                  f"{npu_ops.NO_DESIGN.get(code, 'nothing to compile')}. See "
                  f"`STREAM_ONLY` in `tools/lib/npu_ops.py`.")
                a("")
            elif status == "honours":
                a(f"Design directory: `{design}/`, built automatically by the "
                  f"exporter for any target of this architecture.")
                a("")
    a("## Where this is checked")
    a("")
    a("- `tools/lib/npu_ops.py` -- the registry. The exporter builds from it.")
    a("- `runtime/include/common/npu_ops_flag.hpp` -- the runtime's second copy")
    a("  of the eight codes and their long names. Neither side can include the")
    a("  other, so each points at the other, and the failure mode of the")
    a("  duplication is a refusal by name on both sides -- never a wrong number.")
    a("- `tools/verify/verify_npu_op_matrix.py` -- asserts this file is exactly")
    a("  what `tools/gen_npu_ops_doc.py` prints, asserts the 40-cell count and")
    a("  the per-status tally, and asserts the runtime accepts or refuses each")
    a("  cell's codes the way this document says.")
    a("")
    return "\n".join(L)


# No leading indent here: the caller prints these lines with its own two-space
# prefix, and a second one here prints the pair misaligned by two columns.
def check_one(path, text) -> bool:
    if not path.exists():
        print(f"{path.name} does not exist; run tools/gen_npu_ops_doc.py")
        return False
    cur = path.read_text()
    if cur == text:
        print(f"ok    {path.name} matches tools/lib/npu_ops.py")
        return True
    print(f"{path.name} is stale; tools/gen_npu_ops_doc.py prints:")
    for line in difflib.unified_diff(cur.splitlines(), text.splitlines(),
                                     path.name, "generated", lineterm="", n=1):
        print("  " + line)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 and print the diff if either is "
                         "stale")
    ns = ap.parse_args()
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