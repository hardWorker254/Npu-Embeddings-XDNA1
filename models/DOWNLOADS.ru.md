# Откуда берутся файлы моделей

<!-- СГЕНЕРИРОВАНО tools/gen_downloads_doc.py из
     tools/data/model_sources.json. Правьте ТОТ файл, не этот. Генератор
     отказывается писать этот файл, когда два источника расходятся с
     models/<name>/CHECKPOINT.json. -->

Все модели этого дерева, откуда их файлы и — что существенно — какие из них
можно скачать как ONNX-экспорт.

Из 18 моделей **8 публикуют ONNX-экспорт, который можно скачать**, и
**10 — нет**. Это не решение этого дерева, это то, что есть в
репозиториях наверху. У тех, у кого нет, нет каталога `onnx/` вовсе, и
запрос возвращает 404. Вот настоящая причина того, что в
[BUILD.md §2.2](../BUILD.md) написано «веса не скачиваются — вы их
кладёте сами»: для десяти из восемнадцати брать нечего, потому что граф надо
получить экспортом из чекпойнта.

(BUILD.md на английском — перевода в дереве нет.)

Проверено **2026-10-06**. Из скачиваемых экспортов **8 реально
скачаны и захэшированы** и совпали с файлами, на которых построено это дерево.
В таблице нет ни одной строки, которая опиралась бы только на листинг.

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

| модель | репозиторий | файл | размер | sha256 файла | проверено |
| --- | --- | --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | [`onnx/model.onnx`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx) | 90,405,214 | `6fd5d72fe4589f18…` | да — скачано 90,405,214 Б, захэшировано, совпало с локальным файлом |
| `bge-base-en-v1.5` | [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) | [`onnx/model.onnx`](https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx) | 435,811,539 | `9bc579acdba21c25…` | да — скачано 435,811,539 Б, захэшировано, совпало с локальным файлом |
| `bge-large-en-v1.5` | [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) | [`onnx/model.onnx`](https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/onnx/model.onnx) | 1,336,854,281 | `69ed3f810d3b6d13…` | да — скачано 1,336,854,281 Б, захэшировано, совпало с локальным файлом |
| `bge-micro-v2` | [`TaylorAI/bge-micro-v2`](https://huggingface.co/TaylorAI/bge-micro-v2) | [`onnx/model.onnx`](https://huggingface.co/TaylorAI/bge-micro-v2/resolve/main/onnx/model.onnx) | 69,035,106 | `9f705befe60d00ca…` | да — скачано 69,035,106 Б, захэшировано, совпало с локальным файлом |
| `bge-small-en-v1.5` | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | [`onnx/model.onnx`](https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx) | 133,093,490 | `828e1496d7fabb79…` | да — скачано 133,093,490 Б, захэшировано, совпало с локальным файлом |
| `mediapipe-hands` | [`opencv/palm_detection_mediapipe`](https://huggingface.co/opencv/palm_detection_mediapipe) | [`palm_detection_mediapipe_2023feb.onnx`](https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/palm_detection_mediapipe_2023feb.onnx) | 3,905,734 | `78ff51c38496b7fc…` | да — скачано 3,905,734 Б, захэшировано, совпало с локальным файлом |
| `mediapipe-hands` | [`opencv/handpose_estimation_mediapipe`](https://huggingface.co/opencv/handpose_estimation_mediapipe) | [`handpose_estimation_mediapipe_2023feb.onnx`](https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx) | 4,099,621 | `db0898ae717b76b0…` | да — скачано 4,099,621 Б, захэшировано, совпало с локальным файлом |
| `mediapipe-pose` | [`opencv/person_detection_mediapipe`](https://huggingface.co/opencv/person_detection_mediapipe) | [`person_detection_mediapipe_2023mar.onnx`](https://huggingface.co/opencv/person_detection_mediapipe/resolve/main/person_detection_mediapipe_2023mar.onnx) | 11,990,159 | `47fd5599d6fa1760…` | да — скачано 11,990,159 Б, захэшировано, совпало с локальным файлом |
| `mediapipe-pose` | [`opencv/pose_estimation_mediapipe`](https://huggingface.co/opencv/pose_estimation_mediapipe) | [`pose_estimation_mediapipe_2023mar.onnx`](https://huggingface.co/opencv/pose_estimation_mediapipe/resolve/main/pose_estimation_mediapipe_2023mar.onnx) | 5,557,238 | `9d89c599319a18fb…` | да — скачано 5,557,238 Б, захэшировано, совпало с локальным файлом |
| `nomic-embed-text-v1.5` | [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | [`onnx/model.onnx`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx) | 547,310,275 | `147d5aa88c210123…` | да — скачано 547,310,275 Б, захэшировано, совпало с локальным файлом |

## Модели без ONNX-экспорта наверху

Для них `curl` не поможет. Чекпойнт на HuggingFace есть, а граф надо из него
экспортировать. **Экспортера для них в этом дереве нет** —
`reference/fetch_model.py` тянет конфиги и токенизаторы и никогда не граф, —
так что шаг с ONNX — ваш. Это написано здесь, а не заменено выдуманной командой.

| модель | репозиторий | лицензия | чего не хватает |
| --- | --- | --- | --- |
| `embeddinggemma-300m` | [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) | gemma | нет ONNX наверху (проверено в обоих репозиториях) |
| `gte-multilingual-base` | [`Alibaba-NLP/gte-multilingual-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | apache-2.0 | нет каталога `onnx/`, запрос 404 |
| `vit-base-patch16-224` | [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224) | apache-2.0 | нет каталога `onnx/`, запрос 404 |
| `whisper-base` | [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) | apache-2.0 | нет каталога `onnx/`, запрос 404 |
| `whisper-large-v3` | [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | apache-2.0 | нет каталога `onnx/`, запрос 404 |
| `whisper-large-v3-turbo` | [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | mit | нет каталога `onnx/`, запрос 404 |
| `whisper-medium` | [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium) | apache-2.0 | нет вообще `CHECKPOINT.json` — ни репозитория, ни списка файлов, ни хеша; и нет ONNX наверху |
| `whisper-small` | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | apache-2.0 | нет ONNX наверху и хеш не записан (`sha256: null`) |
| `whisper-tiny` | [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) | apache-2.0 | нет каталога `onnx/`, запрос 404 |
| `yolov8n-pose` | **не модель HuggingFace** | ? | нет каталога `onnx/`, запрос 404 |

**Список `file` в `CHECKPOINT.json` — это не список для скачивания.** Шесть
размеров Whisper записывают `onnx/encoder_model.onnx` и
`onnx/decoder_model.onnx`, и **оба 404 наверху**: эти записи называют то, как
назван собственный экспорт этого дерева, а не откуда его брать. Читайте их как
локальные пути, которые надо создать.

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

