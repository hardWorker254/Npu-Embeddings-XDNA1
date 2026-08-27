# NpuEmbeddings -- is the reference XLM-R Unigram tokenizer the same
# function as HuggingFace's? SPDX-License-Identifier: Apache-2.0
#
# Same discipline as tools/verify_tokenizer.py and
# tools/verify_tokenizer_gemma.py: a tokenizer is a pure function from text
# to ids, so the only honest test is running both implementations over the
# same inputs and comparing every id. Here the reference side is
# tools/xlmr_tokenizer_ref.py reading the XLMRTOK1 blob (the executable
# spec the future C++ port will be diffed against), and the ground truth is
# transformers' AutoTokenizer over models/gte-multilingual-base, loaded
# from the local directory with no network.
#
# The corpus is categorized and deliberately adversarial for THIS pipeline:
# CJK without spaces is where Viterbi bugs hide, NFKC-sensitive input is
# where the precompiled charsmap does, decomposed diacritics are where the
# grapheme-cluster walk does, and whitespace/empty/unknown-codepoint inputs
# are where the edges are. Comparison is byte-exact id sequences, special
# tokens included, no truncation or padding on either side. Mismatch
# DETAIL (including the offending text) goes only into the JSON report;
# stdout carries category names and counts.
#
# Deliberate exclusion: literal special-token strings ("<s>", "<mask>",
# ...) -- the reference implements no added-tokens splitter; see the
# KNOWN LIMITATION note in tools/xlmr_tokenizer_ref.py.
#
# Env: .venv-ref (transformers)
# Usage:
#   & ".\.venv-ref\Scripts\python.exe" tools\verify_tokenizer_xlmr.py
#     [--model-dir DIR] [--table PATH] [--out PATH]

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from xlmr_tokenizer_ref import XlmrTokenizer  # noqa: E402

Z = "‍"  # ZWJ
V16 = "️"  # variation selector-16

CORPUS: dict[str, list[str]] = {
    "english": [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "I can't believe it's not butter!",
        "State-of-the-art results on twenty-four benchmarks.",
        "She said: \"meet me at 5 o'clock, don't be late.\"",
        "COVID-19 changed everything in 2020.",
        "https://example.com/path?query=value&x=1",
        "user@example.com sent a re-encoded MP3 file",
        "The NASA/ESA telescope observed exoplanet WASP-96b.",
        "well, THAT was unexpected... wasn't it?",
        "A 3.5mm jack and a USB-C port",
        "co-operate, naive, resume, cafe",
        "It costs $19.99 (plus tax).",
        "Mr. Smith's dog -- a beagle -- barked.",
        "READ THE MANUAL BEFORE OPERATING",
        "the mitochondria is the powerhouse of the cell",
        "antidisestablishmentarianism",
        "a",
        "I",
        "ok",
        "Embedding models map text into dense vector spaces.",
        "Please rate our service from 1 to 5 stars.",
        "The meeting was rescheduled to Tuesday afternoon.",
        "Half of one, six dozen of the other.",
        "New York City subway map, 42nd Street station",
        "He whispered, 'run.'",
        "GDP grew by 2.3% year-over-year.",
        "The API returns JSON with UTF-8 encoding.",
        "quarterly earnings call transcript, Q3 2025",
        "Don't panic; bring a towel.",
    ],
    "norwegian": [
        "blåbærsyltetøy på skiva til frokost",
        "Jeg kjørte gjennom Ålesund i øsende regnvær.",
        "Køen på E6 var lang søndag ettermiddag.",
        "Hun kjøpte færre bøker enn i fjor.",
        "Været i Tromsø er kaldt og vått om høsten.",
        "Brønnøysundregistrene håndterer næringslivets data.",
        "smørbrød med røkelaks og eggerøre",
        "Å lære seg norsk er ikke så vanskelig.",
        "Fjorden lå blank som et speil i går kveld.",
        "Sæterbakken skrev flere mørke romaner.",
        "tjueåtte grader og sol på Sørlandet",
        "Kongeørnen hekker i bratte fjellvegger.",
        "Bæremeisen passer barn opp til femten kilo.",
        "Grøten trenger mer smør, sa bestemor.",
        "Vålerenga møter Rosenborg på Lerkendal.",
        "øl og akevitt til pinnekjøttet",
        "Direktoratet foreslår nye miljøkrav for oppdrettsnæringen.",
        "en øy ytterst i skjærgården",
        "Barnehagen feiret samefolkets dag med joik.",
        "ÆØÅ æøå",
    ],
    "chinese": [
        "我今天去了北京的图书馆借了三本书",
        "机器学习模型需要大量的训练数据",
        "长江是亚洲最长的河流全长约六千三百公里",
        "他昨天晚上在餐厅吃了一碗牛肉面",
        "深圳的科技公司发展速度非常快",
        "中文分词是自然语言处理的基础任务",
        "她每天早上六点起床跑步锻炼身体",
        "这家店的小笼包皮薄馅多汁水丰富",
        "量子计算机的原理和传统计算机完全不同",
        "春节期间高速公路免收通行费",
        "熊猫喜欢吃竹子也会吃一些水果",
        "上海地铁二号线连接浦东机场和虹桥枢纽",
        "考试成绩将于下周五在网上公布",
        "这部电影讲述了一个关于时间旅行的故事",
        "请把窗户关上外面风太大了",
        "繁體中文與簡體中文的轉換並不總是一一對應",
        "香港的茶餐廳供應菠蘿油和絲襪奶茶",
        "词向量把词语映射到高维空间",
        "他用毛笔写了一幅春联贴在门上",
        "云计算平台提供弹性的存储和算力",
    ],
    "japanese": [
        "東京タワーの近くで美味しいラーメンを食べました",
        "機械学習モデルは大量のデータで訓練されます",
        "新幹線は東京と大阪を約二時間半で結びます",
        "彼女は毎朝コーヒーを飲みながら新聞を読む",
        "日本語のトークン化は空白がないので難しい",
        "桜の花が満開になると公園は人でいっぱいです",
        "コンビニで温かいお弁当とお茶を買った",
        "富士山の頂上から見る日の出は格別だった",
        "この辞書には約二十万語が収録されている",
        "電車の中では携帯電話をマナーモードにしてください",
        "彼はギターとピアノの両方を演奏できます",
        "来週の会議は午後三時から始まる予定です",
        "カタカナとひらがなと漢字が混ざった文章",
        "図書館で借りた本を明日返さなければならない",
        "夏祭りで金魚すくいと花火を楽しんだ",
        "スマートフォンのバッテリーがもうすぐ切れそう",
        "駅前の新しいパン屋のクロワッサンは絶品だ",
        "宿題を終えてからゲームをしてもいいですよ",
        "雨が降りそうだから傘を持って行きなさい",
        "ベクトル検索は意味の近い文書を見つける",
    ],
    "korean": [
        "한국어 형태소 분석은 어렵다",
        "서울의 지하철은 매우 편리합니다",
        "김치찌개와 된장찌개 중에 뭘 먹을까",
        "자연어 처리 모델이 문장을 벡터로 바꾼다",
        "주말에 한강 공원에서 자전거를 탔어요",
        "이 책은 도서관에서 빌린 것입니다",
        "비빔밥에는 고추장을 넣어야 제맛이다",
        "내일 아침 일찍 회의가 있어서 먼저 갈게요",
        "부산행 기차표를 두 장 예매했습니다",
        "겨울에는 눈이 많이 내려서 길이 미끄럽다",
        "친구와 함께 영화를 보러 갔다",
        "한글은 세종대왕이 창제한 문자입니다",
        "냉면은 여름에 먹어야 시원하고 맛있다",
        "회사 근처에 새로 생긴 카페가 정말 좋아요",
        "휴대폰 배터리가 거의 다 떨어졌어요",
    ],
    "cyrillic": [
        "Привет, как дела?",
        "Москва не сразу строилась.",
        "Машинное обучение требует больших данных.",
        "Он прочитал роман за одну ночь.",
        "Съешь же ещё этих мягких французских булок.",
        "Библиотека находится рядом с вокзалом.",
        "Вчера шёл дождь, а сегодня светит солнце.",
        "Эти векторы имеют одинаковую длину.",
        "Українська мова має свої особливі літери: ї, є, ґ.",
        "Дніпро — одна з найбільших річок Європи.",
        "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "Токенизация — первый шаг обработки текста.",
        "Поезд отправляется с третьего пути в полдень.",
        "Зимой в Сибири бывает минус сорок.",
        "Он говорит по-русски, по-английски и по-немецки.",
    ],
    "arabic": [
        "مرحبا بكم في المكتبة العربية",
        "اللغة العربية تكتب من اليمين إلى اليسار",
        "ذهب الطالب إلى الجامعة صباح اليوم",
        "التعلم الآلي يحتاج إلى بيانات كثيرة",
        "القاهرة أكبر مدينة في العالم العربي",
        "إِنَّ مَعَ الْعُسْرِ يُسْرًا",
        "كَتَبَ الوَلَدُ دَرْسَهُ كامِلاً",
        "شربتُ القهوةَ في المقهى القريبِ",
        "الرياضُ عاصمةُ المملكةِ العربيةِ السعوديةِ",
        "يُطبَخُ الكسكسُ يومَ الجمعةِ في المغربِ",
        "قرأتْ فاطمةُ كتابًا عن تاريخِ الأندلسِ",
        "النموذجُ يحوّلُ النصَّ إلى متجهاتٍ رقميةٍ",
        "سافرنا بالقطار من الدار البيضاء إلى فاس",
        "قواعد النحو العربي دقيقة ومنطقية",
        "قال تعالى في محكم التنزيل",
    ],
    "thai": [
        "ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ",
        "ฉันชอบกินข้าวผัดกับไข่ดาว",
        "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
        "การเรียนรู้ของเครื่องต้องใช้ข้อมูลจำนวนมาก",
        "วันนี้อากาศร้อนมากอุณหภูมิสามสิบแปดองศา",
        "รถไฟฟ้าสายสีเขียววิ่งผ่านสยามสแควร์",
        "ต้มยำกุ้งเป็นอาหารไทยที่มีชื่อเสียง",
        "เขาอ่านหนังสือในห้องสมุดทุกวันเสาร์",
        "น้ำตกที่เชียงใหม่สวยงามมากในฤดูฝน",
        "โมเดลภาษาแปลงข้อความเป็นเวกเตอร์",
        "ยายปลูกมะม่วงไว้หลังบ้านสามต้น",
        "พรุ่งนี้เช้ามีประชุมสำคัญที่บริษัท",
    ],
    "hindi": [
        "मशीन लर्निंग को बहुत सारे डेटा की आवश्यकता होती है",
        "मैं कल दिल्ली से मुंबई जा रहा हूँ",
        "हिन्दी देवनागरी लिपि में लिखी जाती है",
        "उसने पुस्तकालय से तीन किताबें लीं",
        "गंगा भारत की सबसे लंबी नदी है",
        "आज मौसम बहुत सुहावना है",
        "क्या आप मेरी मदद कर सकते हैं?",
        "विद्यार्थी परीक्षा की तैयारी कर रहे हैं",
        "संस्कृत के श्लोक कण्ठस्थ करना कठिन है",
        "प्रधानमंत्री ने नई योजना की घोषणा की",
    ],
    "hebrew_greek": [
        "שלום עולם, מה שלומך היום?",
        "העברית נכתבת מימין לשמאל",
        "בְּרֵאשִׁית בָּרָא אֱלֹהִים",
        "המודל ממיר טקסט לווקטורים",
        "תל אביב יושבת על חוף הים התיכון",
        "Η γρήγορη καφέ αλεπού πηδά πάνω από τον τεμπέλη σκύλο.",
        "Τα διανύσματα έχουν μήκος και κατεύθυνση.",
        "Ο Παρθενώνας βρίσκεται στην Ακρόπολη των Αθηνών.",
        "Η μηχανική μάθηση χρειάζεται πολλά δεδομένα.",
        "αβγδεζηθικλμνξοπρστυφχψω",
    ],
    "vietnamese_turkish": [
        "Tôi thích ăn phở bò vào buổi sáng.",
        "Hà Nội là thủ đô của Việt Nam.",
        "Máy học cần rất nhiều dữ liệu huấn luyện.",
        "Cô ấy đọc sách ở thư viện mỗi cuối tuần.",
        "Tiếng Việt có sáu thanh điệu khác nhau.",
        "Đường phố Sài Gòn đông đúc vào giờ cao điểm.",
        "Chúng tôi đã đặt vé máy bay đi Đà Nẵng.",
        "İstanbul Boğazı iki kıtayı birbirine bağlar.",
        "Türkçede büyük İ ve küçük ı farklı harflerdir.",
        "Yapay öğrenme çok veri gerektirir.",
        "Ankara'da kış ayları soğuk geçer.",
        "Çocuklar bahçede top oynuyor.",
        "Gözlerin gözlerime değince felaketim olurdu ağlardım.",
    ],
    "emoji": [
        "I love pizza 🍕",
        "🍕",
        "👍",
        "👍🏽",
        "thumbs up 👍🏿 dark skin tone",
        "👨" + Z + "👩" + Z + "👧" + Z + "👦 family",
        "woman firefighter 👩" + Z + "🚒",
        "🏳️" + Z + "🌈 rainbow flag",
        "🇳🇴 Norway and 🇯🇵 Japan flags",
        "🇳🇴🇯🇵 adjacent flags",
        "keycap 1️⃣ and asterisk *️⃣",
        "heart ❤ vs ❤" + V16 + " with VS16",
        "🤷" + Z + "♀" + V16 + " woman shrugging",
        "mixed 🎉 party 🎊 emoji 🎈 text",
        "😀😃😄😁😆😅🤣😂",
    ],
    "mixed_script": [
        "iPhone15を買いました",
        "他说hello然后就走了",
        "Das Café am Königsplatz öffnet um 8 Uhr.",
        "СССР launched Спутник in 1957",
        "email是电子邮件的意思",
        "K-pop 아이돌 그룹이 Billboard 차트에 올랐다",
        "π의 값은 약 3.14159이다",
        "Wi-Fiのパスワードは12345678です",
        "αβγ meets abc and абв",
        "東京2020オリンピック",
        "C++和Python都是编程语言",
        "ملف PDF باللغة العربية",
        "ราคา 500 บาท ครับ",
        "Ñandú corre por la pampa Argentina.",
        "résumé.docx と 履歴書.pdf",
    ],
    "nfkc": [
        "ＡＢＣ full-width latin",
        "ｈｅｌｌｏ　ｗｏｒｌｄ",
        "１２３４５ full-width digits",
        "ﬁ ﬂ ﬀ ﬃ ﬄ ligatures",
        "oﬃce ﬂoor ﬁnal",
        "x² + y³ = z⁴",
        "H₂O and CO₂",
        "① ② ③ ⑩ circled",
        "Ⅳ Ⅸ Ⅻ roman numerals",
        "ⅳ ⅸ ⅻ lowercase roman",
        "㎒ ㎑ ㎓ frequency units",
        "㎞ ㎡ ㎥ ㌔",
        "café decomposed: café",
        "Å ring above decomposed",
        "ȩ́ stacked combining marks",
        "ẛ̣ long s dot above plus dot below",
        "ǆ ǅ Ǆ digraphs",
        "ｶﾀｶﾅ half-width katakana",
        "ｶﾞｷﾞｸﾞ half-width with dakuten",
        "…ellipsis and ‥two dots",
        "﷼ rial sign",
        "ﷺ expands to a whole phrase",
        "µ micro vs μ mu",
        "Ω ohm vs Ω omega",
        "Ａｐｐｌｅ ｖｓ Apple",
        "½ ⅓ ¼ ⅛ fractions",
        "№5 numero sign",
        "™ and © and ®",
        "ﻻ lam-alef ligature",
        "ﬆ ﬅ st ligatures",
    ],
    "whitespace": [
        "",
        " ",
        "   ",
        "\t",
        "hello world",
        " hello world",
        "hello world ",
        "  hello   world  ",
        "\thello\tworld\t",
        "hello\nworld",
        "hello\r\nworld",
        "a b nbsp between",
        "thin space",
        "ideographic　space",
        "　",
        "em space and en space",
        "zero​width space between",
        "ctrlseparator",
    ],
    "long": [
        "supercalifragilisticexpialidocious" * 10,
        "a" * 300,
        "ab" * 200,
        "很" * 150,
        "อ" * 120,
        ("Donaudampfschifffahrtsgesellschaftskapitaen" * 6),
    ],
    "unk": [
        "\U00012031",
        "\U00012031\U00012032",
        "x\U00012031\U00012032y",
        "cuneiform \U00012000 sign",
        "\U0001D000\U0001D001 byzantine music",
        "linear b \U00010000\U00010001 tablets",
        "deseret \U00010400\U00010428 letters",
        " private use",
        "word\U00012031word",
        "\U00012031 \U00012032 spaced unknowns",
        "mixed известно \U00012031 неизвестно",
        "\U000E0041\U000E0042 tag characters",
    ],
    "punct_numbers": [
        "!@#$%^&*()_+-=[]{}|;:,.<>?/~`",
        "“smart quotes” and ‘single’",
        "em—dash and en–dash and minus−sign",
        "3.14159 and 2,718.28 and 1e-10",
        "50% off, was $100, now €50, ¥5000, £45, ₹999, ₩10000",
        "+47 22 44 66 88",
        "2026-08-27T14:30:00Z",
        "IPv4 192.168.0.1 port 8080",
        "the answer is 42",
        "٠١٢٣٤٥٦٧٨٩ arabic-indic digits",
        "١٢٣ مع نص عربي",
        "देवनागरी अंक ०१२३४५६७८९",
        "→ ← ↑ ↓ ⇒ ⇔ arrows",
        "≤ ≥ ≠ ≈ ∞ ∑ ∏ √ ∫ math",
        "«guillemets» and „low quotes”",
    ],
}


# Programmatic NFD tier: decomposed forms of sentences from the scripts
# where canonical decomposition actually changes the codepoints (Hangul
# syllables explode into jamo, Vietnamese/Norwegian diacritics split into
# base + combining marks). This is where the charsmap's multi-char
# recomposition keys and the grapheme-cluster walk earn their keep.
import unicodedata as _ud

CORPUS["nfd_decomposed"] = sorted({
    _ud.normalize("NFD", t)
    for cat in ("korean", "vietnamese_turkish", "norwegian", "hindi",
                "hebrew_greek", "arabic")
    for t in CORPUS[cat]
    if _ud.normalize("NFD", t) != t
})


def run_cli(cli: str, table: str, texts: list[str]) -> list[list[int]]:
    """Drive the C++ CLI (tokenizer_xlmr_cli.exe) over `texts`, one escaped
    line each -- the mirror image of the CLI's documented unescape (two
    corpus entries carry embedded newlines, which a line protocol cannot
    ship raw). Returns one id list per text."""
    import subprocess
    import tempfile

    def esc(t: str) -> str:
        return t.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")

    with tempfile.TemporaryDirectory(prefix="xlmrtokverify_") as td:
        inp = Path(td) / "corpus.txt"
        with inp.open("w", encoding="utf-8", newline="\n") as f:
            for t in texts:
                f.write(esc(t) + "\n")
        r = subprocess.run([cli, table, str(inp)], capture_output=True,
                           text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"C++ CLI failed ({r.returncode}):\n{r.stderr}")
    lines = r.stdout.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    # keep exact line count -- an empty input still emits a line ("0 2")
    if len(lines) != len(texts):
        raise SystemExit(f"C++ CLI produced {len(lines)} lines for "
                         f"{len(texts)} texts")
    return [[int(x) for x in ln.split()] for ln in lines]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(
        REPO / "models" / "gte-multilingual-base"))
    ap.add_argument("--table", default=str(
        REPO / "models" / "gte-multilingual-base" / "xlmr_tokenizer.bin"))
    ap.add_argument("--cli", default=None,
                    help="path to xlmr_tok_cli.exe; when given, the C++ "
                         "implementation is verified against BOTH HuggingFace "
                         "and the Python reference (three-way), e.g. "
                         "runtime\\build\\xlmr_tok_cli.exe")
    ap.add_argument("--out", default=str(
        REPO / "tasks" / "0127-t52-unigram-generator"
        / "verify_tokenizer_xlmr.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(args.model_dir)
    ref = XlmrTokenizer(args.table)

    items = [(cat, text) for cat, texts in CORPUS.items() for text in texts]
    cli_ids: list[list[int]] | None = None
    if args.cli:
        cli_ids = run_cli(args.cli, args.table, [t for _, t in items])

    n_ok = n_total = 0
    n_cli_hf_ok = n_cli_ref_ok = 0
    per_cat: dict[str, list[int]] = {c: [0, 0, len(ts)]
                                     for c, ts in CORPUS.items()}
    mismatches: list[dict] = []

    for idx, (cat, text) in enumerate(items):
        want = [int(v) for v in hf(text)["input_ids"]]
        got = ref.encode(text)
        n_total += 1
        row_bad = False
        if got == want:
            n_ok += 1
            per_cat[cat][0] += 1
        else:
            row_bad = True
        if cli_ids is not None:
            cpp = cli_ids[idx]
            if cpp == want:
                n_cli_hf_ok += 1
                per_cat[cat][1] += 1
            else:
                row_bad = True
            if cpp == got:
                n_cli_ref_ok += 1
        if row_bad:
            i = next((k for k in range(max(len(got), len(want)))
                      if k >= len(got) or k >= len(want)
                      or got[k] != want[k]), 0)
            mismatches.append({
                "category": cat,
                "text": text,
                "first_diff_index": i,
                "hf": want,
                "ref": got,
                **({"cpp": cli_ids[idx]} if cli_ids is not None else {}),
            })

    width = max(len(c) for c in CORPUS)
    for cat, (ok, cli_ok, tot) in per_cat.items():
        good = ok == tot and (cli_ids is None or cli_ok == tot)
        mark = "ok " if good else "FAIL"
        cli_col = f"  cpp {cli_ok}/{tot}" if cli_ids is not None else ""
        print(f"  {mark} {cat:<{width}} {ok}/{tot}{cli_col}")
    print(f"tokenizer verification: ref-vs-HF {n_ok}/{n_total} exact "
          f"({n_ok / n_total * 100:.2f}%)  "
          f"(special tokens included, no truncation, no padding)")
    if cli_ids is not None:
        print(f"  C++ CLI vs HuggingFace:       {n_cli_hf_ok}/{n_total} exact")
        print(f"  C++ CLI vs Python reference:  {n_cli_ref_ok}/{n_total} exact")
    if mismatches:
        print(f"  {len(mismatches)} mismatches -- detail (including the "
              f"offending texts) is in the JSON report only")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "verification",
        "model": "gte-multilingual-base",
        "reference": "tools/xlmr_tokenizer_ref.py over XLMRTOK1 blob",
        "cli": args.cli,
        "n_total": n_total,
        "n_exact": n_ok,
        "agreement": n_ok / n_total,
        **({"n_cli_vs_hf_exact": n_cli_hf_ok,
            "n_cli_vs_ref_exact": n_cli_ref_ok} if cli_ids is not None else {}),
        "per_category": {c: {"ok": ok, "total": tot,
                             **({"cpp_ok": cli_ok}
                                if cli_ids is not None else {})}
                         for c, (ok, cli_ok, tot) in per_cat.items()},
        "mismatches": mismatches,
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {out}")
    ok = n_ok == n_total and (cli_ids is None or
                              (n_cli_hf_ok == n_total and
                               n_cli_ref_ok == n_total))
    print("PASS -- byte-for-byte identical to HuggingFace" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
