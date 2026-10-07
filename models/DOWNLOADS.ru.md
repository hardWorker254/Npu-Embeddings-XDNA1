# Откуда берутся файлы моделей

<!-- СГЕНЕРИРОВАНО tools/gen_downloads_doc.py из
     tools/data/model_sources.json. Правьте ТОТ файл, не этот. Генератор
     отказывается писать этот файл, когда два источника расходятся с
     models/<name>/CHECKPOINT.json. -->

Все модели этого дерева, откуда их файлы и — что существенно — какие из них
можно скачать как ONNX-экспорт.

Из 18 моделей **18 публикуют ONNX-экспорт, который можно скачать**, и
**0 — нет**. Это не решение этого дерева, это то, что есть в
репозиториях наверху. У тех, у кого нет, нет каталога `onnx/` вовсе, и
запрос возвращает 404. Вот настоящая причина того, что в
[BUILD.md §2.2](../BUILD.md) написано «веса не скачиваются — вы их
кладёте сами»: для десяти из восемнадцати брать нечего, потому что граф надо
получить экспортом из чекпойнта.

(BUILD.md на английском — перевода в дереве нет.)

Проверено **2026-10-06**. Файлы в **9 строках скачаны и захэшированы**;
остальные 9 — только листинг по размеру из `HEAD`, и строка говорит,
какая именно. «Захэшировано» — не то же самое, что «совпало»: у **`whisper-tiny`, `yolov8n-pose`**
файл скачан, упакован, и **совпал контейнер** — тот, что собрало это дерево. Для
зеркала это утверждение сильнее и другое, поэтому такие строки говорят «упакованный результат тот же», а не «совпало».

## Два разных хеша, и путаница между ними не проверяет ничего

`CHECKPOINT.json` использует **два несовместимых соглашения** для `sha256`,
и какое из них у модели — свойство самой модели:

| `digest_kind` | что это | у кого |
| --- | --- | --- |
| `model_digest` | `sha256` над `(имя файла, NUL, sha256-файла, LF)` для каждого исходного файла ONNX, в фиксированном порядке — `tools/lib/onnx_weights.py:model_digest()`. **Не** sha256 ни одного отдельного файла. | у всех трансформерных моделей |
| `file_sha256` | **список** обычных sha256, по одному на запись в `file` | у двух моделей MediaPipe |

Оба измерены, пока писался этот документ. `bge-small-en-v1.5` записывает
`1fd2e85d...` — это его `model_digest`; sha256 её настоящего
`onnx/model.onnx` равен `828e1496...`. Сверка скачанного с записанным пином по
неверному соглашению fails на корректном файле и не проверяет ничего.

## Скачиваемые ONNX-экспорты

| модель | репозиторий | чей | файл | размер | sha256 файла | проверено |
| --- | --- | --- | --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | свой | [`onnx/model.onnx`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx) | 90,405,214 | `6fd5d72fe4589f18…` | да — скачано 90,405,214 Б, захэшировано, совпало с локальным файлом |
| `bge-base-en-v1.5` | [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) | свой | [`onnx/model.onnx`](https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx) | 435,811,539 | `9bc579acdba21c25…` | да — скачано 435,811,539 Б, захэшировано, совпало с локальным файлом |
| `bge-large-en-v1.5` | [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) | свой | [`onnx/model.onnx`](https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/onnx/model.onnx) | 1,336,854,281 | `69ed3f810d3b6d13…` | да — скачано 1,336,854,281 Б, захэшировано, совпало с локальным файлом |
| `bge-micro-v2` | [`TaylorAI/bge-micro-v2`](https://huggingface.co/TaylorAI/bge-micro-v2) | свой | [`onnx/model.onnx`](https://huggingface.co/TaylorAI/bge-micro-v2/resolve/main/onnx/model.onnx) | 69,035,106 | `9f705befe60d00ca…` | да — скачано 69,035,106 Б, захэшировано, совпало с локальным файлом |
| `bge-small-en-v1.5` | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | свой | [`onnx/model.onnx`](https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx) | 133,093,490 | `828e1496d7fabb79…` | да — скачано 133,093,490 Б, захэшировано, совпало с локальным файлом |
| `embeddinggemma-300m` | [`onnx-community/embeddinggemma-300m-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) | **зеркало** | [`onnx/model.onnx`](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model.onnx) | 479,932 | — | **только листинг, байты не доказаны** |
| `embeddinggemma-300m` | [`onnx-community/embeddinggemma-300m-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) | **зеркало** | [`onnx/model.onnx_data`](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model.onnx_data) | 1,234,521,088 | — | **только листинг, байты не доказаны** |
| `gte-multilingual-base` | [`onnx-community/gte-multilingual-base`](https://huggingface.co/onnx-community/gte-multilingual-base) | **зеркало** | [`onnx/model.onnx`](https://huggingface.co/onnx-community/gte-multilingual-base/resolve/main/onnx/model.onnx) | 1,255,502,649 | — | **только листинг, байты не доказаны** |
| `mediapipe-hands` | [`opencv/palm_detection_mediapipe`](https://huggingface.co/opencv/palm_detection_mediapipe) | свой | [`palm_detection_mediapipe_2023feb.onnx`](https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/palm_detection_mediapipe_2023feb.onnx) | 3,905,734 | `78ff51c38496b7fc…` | да — скачано 3,905,734 Б, захэшировано, совпало с локальным файлом |
| `mediapipe-hands` | [`opencv/handpose_estimation_mediapipe`](https://huggingface.co/opencv/handpose_estimation_mediapipe) | свой | [`handpose_estimation_mediapipe_2023feb.onnx`](https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx) | 4,099,621 | `db0898ae717b76b0…` | да — скачано 4,099,621 Б, захэшировано, совпало с локальным файлом |
| `mediapipe-pose` | [`opencv/person_detection_mediapipe`](https://huggingface.co/opencv/person_detection_mediapipe) | свой | [`person_detection_mediapipe_2023mar.onnx`](https://huggingface.co/opencv/person_detection_mediapipe/resolve/main/person_detection_mediapipe_2023mar.onnx) | 11,990,159 | `47fd5599d6fa1760…` | да — скачано 11,990,159 Б, захэшировано, совпало с локальным файлом |
| `mediapipe-pose` | [`opencv/pose_estimation_mediapipe`](https://huggingface.co/opencv/pose_estimation_mediapipe) | свой | [`pose_estimation_mediapipe_2023mar.onnx`](https://huggingface.co/opencv/pose_estimation_mediapipe/resolve/main/pose_estimation_mediapipe_2023mar.onnx) | 5,557,238 | `9d89c599319a18fb…` | да — скачано 5,557,238 Б, захэшировано, совпало с локальным файлом |
| `nomic-embed-text-v1.5` | [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | свой | [`onnx/model.onnx`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx) | 547,310,275 | `147d5aa88c210123…` | да — скачано 547,310,275 Б, захэшировано, совпало с локальным файлом |
| `vit-base-patch16-224` | [`onnx-community/vit-base-patch16-224-ONNX`](https://huggingface.co/onnx-community/vit-base-patch16-224-ONNX) | **зеркало** | [`onnx/model.onnx`](https://huggingface.co/onnx-community/vit-base-patch16-224-ONNX/resolve/main/onnx/model.onnx) | 346,471,546 | — | **только листинг, байты не доказаны** |
| `whisper-base` | [`onnx-community/whisper-base`](https://huggingface.co/onnx-community/whisper-base) | **зеркало** | [`onnx/encoder_model.onnx`](https://huggingface.co/onnx-community/whisper-base/resolve/main/onnx/encoder_model.onnx) | 82,468,078 | — | **только листинг, байты не доказаны** |
| `whisper-base` | [`onnx-community/whisper-base`](https://huggingface.co/onnx-community/whisper-base) | **зеркало** | [`onnx/decoder_model.onnx`](https://huggingface.co/onnx-community/whisper-base/resolve/main/onnx/decoder_model.onnx) | 208,289,724 | — | **только листинг, байты не доказаны** |
| `whisper-large-v3` | [`onnx-community/whisper-large-v3-ONNX`](https://huggingface.co/onnx-community/whisper-large-v3-ONNX) | **зеркало** | [`onnx/encoder_model.onnx`](https://huggingface.co/onnx-community/whisper-large-v3-ONNX/resolve/main/onnx/encoder_model.onnx) | 412,412 | — | **только листинг, байты не доказаны** |
| `whisper-large-v3` | [`onnx-community/whisper-large-v3-ONNX`](https://huggingface.co/onnx-community/whisper-large-v3-ONNX) | **зеркало** | [`onnx/encoder_model.onnx_data`](https://huggingface.co/onnx-community/whisper-large-v3-ONNX/resolve/main/onnx/encoder_model.onnx_data) | 2,547,875,840 | — | **только листинг, байты не доказаны** |
| `whisper-large-v3` | [`onnx-community/whisper-large-v3-ONNX`](https://huggingface.co/onnx-community/whisper-large-v3-ONNX) | **зеркало** | [`onnx/decoder_model.onnx`](https://huggingface.co/onnx-community/whisper-large-v3-ONNX/resolve/main/onnx/decoder_model.onnx) | 1,141,794 | — | **только листинг, байты не доказаны** |
| `whisper-large-v3` | [`onnx-community/whisper-large-v3-ONNX`](https://huggingface.co/onnx-community/whisper-large-v3-ONNX) | **зеркало** | [`onnx/decoder_model.onnx_data`](https://huggingface.co/onnx-community/whisper-large-v3-ONNX/resolve/main/onnx/decoder_model.onnx_data) | 3,626,086,400 | — | **только листинг, байты не доказаны** |
| `whisper-large-v3-turbo` | [`onnx-community/whisper-large-v3-turbo`](https://huggingface.co/onnx-community/whisper-large-v3-turbo) | **зеркало** | [`onnx/encoder_model.onnx`](https://huggingface.co/onnx-community/whisper-large-v3-turbo/resolve/main/onnx/encoder_model.onnx) | 439,254 | — | **только листинг, байты не доказаны** |
| `whisper-large-v3-turbo` | [`onnx-community/whisper-large-v3-turbo`](https://huggingface.co/onnx-community/whisper-large-v3-turbo) | **зеркало** | [`onnx/encoder_model.onnx_data`](https://huggingface.co/onnx-community/whisper-large-v3-turbo/resolve/main/onnx/encoder_model.onnx_data) | 2,547,875,840 | — | **только листинг, байты не доказаны** |
| `whisper-large-v3-turbo` | [`onnx-community/whisper-large-v3-turbo`](https://huggingface.co/onnx-community/whisper-large-v3-turbo) | **зеркало** | [`onnx/decoder_model.onnx`](https://huggingface.co/onnx-community/whisper-large-v3-turbo/resolve/main/onnx/decoder_model.onnx) | 687,820,062 | — | **только листинг, байты не доказаны** |
| `whisper-medium` | [`flackzz/whisper-medium-ONNX`](https://huggingface.co/flackzz/whisper-medium-ONNX) | **зеркало** | [`onnx/encoder_model.onnx`](https://huggingface.co/flackzz/whisper-medium-ONNX/resolve/main/onnx/encoder_model.onnx) | 1,229,148,954 | — | **только листинг, байты не доказаны** |
| `whisper-medium` | [`flackzz/whisper-medium-ONNX`](https://huggingface.co/flackzz/whisper-medium-ONNX) | **зеркало** | [`onnx/decoder_model.onnx`](https://huggingface.co/flackzz/whisper-medium-ONNX/resolve/main/onnx/decoder_model.onnx) | 1,827,381,748 | — | **только листинг, байты не доказаны** |
| `whisper-small` | [`onnx-community/whisper-small`](https://huggingface.co/onnx-community/whisper-small) | **зеркало** | [`onnx/encoder_model.onnx`](https://huggingface.co/onnx-community/whisper-small/resolve/main/onnx/encoder_model.onnx) | 352,825,870 | — | **только листинг, байты не доказаны** |
| `whisper-small` | [`onnx-community/whisper-small`](https://huggingface.co/onnx-community/whisper-small) | **зеркало** | [`onnx/decoder_model.onnx`](https://huggingface.co/onnx-community/whisper-small/resolve/main/onnx/decoder_model.onnx) | 614,865,004 | — | **только листинг, байты не доказаны** |
| `whisper-tiny` | [`onnx-community/whisper-tiny-ONNX`](https://huggingface.co/onnx-community/whisper-tiny-ONNX) | **зеркало** | [`onnx/encoder_model.onnx`](https://huggingface.co/onnx-community/whisper-tiny-ONNX/resolve/main/onnx/encoder_model.onnx) | 32,883,618 | `8dd994fe489eaa52…` | да — скачано 32,883,618 Б, захэшировано, **упакованный результат тот же** |
| `whisper-tiny` | [`onnx-community/whisper-tiny-ONNX`](https://huggingface.co/onnx-community/whisper-tiny-ONNX) | **зеркало** | [`onnx/decoder_model.onnx`](https://huggingface.co/onnx-community/whisper-tiny-ONNX/resolve/main/onnx/decoder_model.onnx) | 118,352,985 | `7e844cce0ac74eda…` | да — скачано 118,352,985 Б, захэшировано, **упакованный результат тот же** |
| `yolov8n-pose` | [`Xenova/yolov8-pose-onnx`](https://huggingface.co/Xenova/yolov8-pose-onnx) | **зеркало** | [`yolov8n-pose.onnx`](https://huggingface.co/Xenova/yolov8-pose-onnx/resolve/main/yolov8n-pose.onnx) | 13,484,153 | `04f6d2416266f2ab…` | **только листинг, байты не доказаны** |

## Зеркала и чего они стоят

У десяти моделей граф лежит в чужом репозитории. Прежде чем им пользоваться,
стоит знать три вещи, и очевидна только первая.

**1. ONNX из зеркала НЕ удовлетворит уже записанный пин.** Это другая
сериализация тех же весов, поэтому digest из `CHECKPOINT.json` не совпадёт с
скачанным. Измерено на whisper-tiny:

| | экспорт этого дерева | onnx-community |
| --- | --- | --- |
| sha256 `encoder_model.onnx` | `6642befb…` | `8dd994fe…` |
| sha256 `decoder_model.onnx` | `ab79e3f2…` | `7e844cce…` |
| `model_digest` | `eb6a1b7f…` | `55ace44d…` |
| **вся область весов контейнера** | `9fd75f76…` | **`9fd75f76…`** |

Один и тот же контейнер, побайтово, из разных входных файлов. Так что если
проверка пина падает на зеркале, правильное действие — **перезаписать пин по
файлу, который вы реально положили**. Не править записанное значение, пока не
совпадёт, и не считать зеркало неверным из-за разницы хешей.

**2. У зеркал в основном нет лицензии.** Тег `license:` есть только у двух
`opencv/*` MediaPipe и у `Xenova/yolov8-pose-onnx` (`agpl-3.0`). У всех
`onnx-community/*` его **нет вовсе**, поэтому лицензия исходного репозитория
перенесена в таблицу **как утверждение**, с пометкой. Репозиторий без тега не
становится от этого apache-2.0, и тому, кому лицензия нужна точно, читать
условия исходного репозитория.

**3. `whisper-medium` — не onnx-community**, а `flackzz/whisper-medium-ONNX`.
Остальные пять размеров Whisper на onnx-community, так что выводить доверие из
закономерности не от чего, и это единственное зеркало в таблице без
измеренного утверждения.

| модель | репозиторий графа | лицензия и откуда это утверждение |
| --- | --- | --- |
| `embeddinggemma-300m` | [`onnx-community/embeddinggemma-300m-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) | gemma — the MIRROR's license: tag -- and it agrees with Google's, so the Gemma terms still apply to this mirror |
| `gte-multilingual-base` | [`onnx-community/gte-multilingual-base`](https://huggingface.co/onnx-community/gte-multilingual-base) | apache-2.0 (asserted, NOT stated) — the MIRROR carries NO license: tag. The original repository's tag says apache-2.0 and this document carries that forward as an assertion, which is … |
| `vit-base-patch16-224` | [`onnx-community/vit-base-patch16-224-ONNX`](https://huggingface.co/onnx-community/vit-base-patch16-224-ONNX) | apache-2.0 (asserted, NOT stated) — the MIRROR carries NO license: tag; the original repository's tag says apache-2.0 |
| `whisper-base` | [`onnx-community/whisper-base`](https://huggingface.co/onnx-community/whisper-base) | apache-2.0 (asserted, NOT stated) — the MIRROR carries NO license: tag; the original openai/whisper-* tag says apache-2.0 |
| `whisper-large-v3` | [`onnx-community/whisper-large-v3-ONNX`](https://huggingface.co/onnx-community/whisper-large-v3-ONNX) | apache-2.0 (asserted, NOT stated) — the MIRROR carries NO license: tag; the original openai/whisper-large-v3 tag says apache-2.0 |
| `whisper-large-v3-turbo` | [`onnx-community/whisper-large-v3-turbo`](https://huggingface.co/onnx-community/whisper-large-v3-turbo) | mit (asserted, NOT stated) — the MIRROR carries NO license: tag; the original repository's tag says mit, and this is the one Whisper size that is not apache-2.0 |
| `whisper-medium` | [`flackzz/whisper-medium-ONNX`](https://huggingface.co/flackzz/whisper-medium-ONNX) | apache-2.0 (asserted, NOT stated) — the MIRROR carries NO license: tag; the original openai/whisper-* tag says apache-2.0 |
| `whisper-small` | [`onnx-community/whisper-small`](https://huggingface.co/onnx-community/whisper-small) | apache-2.0 (asserted, NOT stated) — the MIRROR carries NO license: tag; the original openai/whisper-* tag says apache-2.0 |
| `whisper-tiny` | [`onnx-community/whisper-tiny-ONNX`](https://huggingface.co/onnx-community/whisper-tiny-ONNX) | apache-2.0 (asserted, NOT stated) — the MIRROR carries NO license: tag; the original openai/whisper-* tag says apache-2.0 |
| `yolov8n-pose` | [`Xenova/yolov8-pose-onnx`](https://huggingface.co/Xenova/yolov8-pose-onnx) | agpl-3.0 — the MIRROR states agpl-3.0, and so does Ultralytics' own release |

### Одна из этих ссылок неверна на одно слово

`https://huggingface.co/onnx-community/whisper-large-v3` отдаёт **HTTP 401** и на список модели, и на любой
файл. Рабочее имя — то же самое **с суффиксом `-ONNX`**, который носят
`whisper-tiny-ONNX` и `vit-base-patch16-224-ONNX`; оно отвечает 200 и
содержит все четыре файла `whisper-large-v3`. Записано потому, что разница — одно
слово с дефисом и читается как опечатка, а не как 401.

**Чем зеркало НЕ является.** Ничто здесь не говорит, что зеркало взаимозаменяемо
с экспортом самой модели. Говорится, что для двух проверенных строк
упакованный результат совпал; для остальных восьми утверждение кончается на
*файл есть и форма правильная*. Только у `yolov8n-pose` и `whisper-tiny` есть
измеренное утверждение, и строки говорят, у каких именно.

## Два репозитория на одну модель: `embeddinggemma-300m`

`models/embeddinggemma-300m/CHECKPOINT.json` записывает
`unsloth/embeddinggemma-300m`, а каталог в `runtime/src/common/hub.cpp`
тянет `google/embeddinggemma-300m`. **Это не взаимозаменяемо:** репозиторий
Google гейтед — нужна принятая лицензия и заголовок `Authorization: Bearer`, и
`hub.cpp` fail'ится закрыто, а не берёт зеркало. Копия unsloth не гейтед и веси
те же веса, так что читатель по `CHECKPOINT.json` и читатель `serve` тянут из
разных мест, и токен нужен только одному. Оба названы в таблице; ни один не
выдаётся за другой.

### 2 модели без `CHECKPOINT.json`

`yolov8n-pose` и `whisper-medium` — два каталога в `models/`, где нет
`CHECKPOINT.json`. Для `whisper-medium` это значит, что модель здесь собрана и
работает вообще без записанного происхождения: ни строки о репозитории, ни
списка файлов, ни хеша. Для `yolov8n-pose` — что дерево никогда не записывало,
откуда взялся ONNX: Ultralytics публикует несколько экспортов `yolov8n-pose`, и
сказать какой — утверждение, требующее доказательства.

Написать эти два `CHECKPOINT.json` — единственная дыра в этом документе,
которую не закроет никакая ссылка: не хватает локальной записи, а не удалённого
файла.

