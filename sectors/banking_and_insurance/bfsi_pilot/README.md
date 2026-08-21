# BFSI pilot — цифровий двійник страхової + AI-агент

Робочий пілот на **одній GPU 16 GB** (моделі використовуються послідовно):
Postgres+pgvector+MinIO як «двійник» страхової, vision-інджест сканів,
агент фахівця врегулювання з SQL-інструментами, CLI-чат, інтеграційний
eval на 20 еталонних питаннях. Кожен модуль — **один Python-файл**, без
зайвих абстракцій. Це водночас **темплейт для інших секторів** — див.
останній розділ.

## Структура

Назва скрипта = його роль у пайплайні (запускаються згори вниз):

```
docker-compose.yml       стан: pgvector/pg16 + MinIO (міграції — автоматично)
db/001_core.sql          схема + ролі ingest_rw / agent_ro
create_demo_data.py      створити демо-дані: 12 клієнтів, 10 полісів (PDF),
                         50 клеймів, 20 сканів, red flags, ground_truth.json
add_documents.py         ВАШІ файли: inputs_docs/*.pdf|png|jpg → MinIO + черга
process_documents.py     обробити чергу: VLM-екстракція сканів + чанкінг
                         полісів; --embed — ембединги в pgvector (Стек B)
agent_server.py          бекенд агента: FastAPI /chat + 4 інструменти; --selftest
chat_cli.py              консольний чат до бекенда; /file <path>; --verbose
run_eval.py              оцінка: 20 питань → eval/report.md; --fields — точність полів
inputs_docs/             сюди кладете свої документи для add_documents.py
eval/golden_questions.yaml
```

Inference-джоби (деплой окремо, в `inference_jobs/vllm/`):

| Стек | Джоба | Env для пілота |
|---|---|---|
| A — інджест | [`Qwen2.5-VL-7B-AWQ_banking_and_insurance_docs_min_8gb_vram`](../../../inference_jobs/vllm/Qwen2.5-VL-7B-AWQ_banking_and_insurance_docs_min_8gb_vram/README.md) | `VL_BASE_URL`, `VL_API_KEY` |
| B — чат+ембед | [`Qwen3-8B-AWQ_banking_and_insurance_chat_embed_min_12gb_vram`](../../../inference_jobs/vllm/Qwen3-8B-AWQ_banking_and_insurance_chat_embed_min_12gb_vram/README.md) | `CHAT_BASE_URL`, `CHAT_API_KEY` (embed — та сама джоба, model id `embed`) |

## Повний флоу запуску (runbook)

Логіка проста: на одній 16 GB карті активний **один** стек. Поки живе
Стек A — обробляємо документи; перемкнулись на Стек B — рахуємо
ембединги і спілкуємось з агентом.

### Крок 0. Підготовка (один раз)

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                # локальні DSN вже валідні; ключі моделей — далі
```

### Крок 1. Інфраструктура + демо-дані

```bash
docker compose up -d                # pgvector + MinIO; міграції — на першому старті
python create_demo_data.py
# ✔ ok: "documents pending=30 (20 сканів + 10 полісів)"
```

### Крок 2. Свої документи (опційно, будь-коли)

```bash
# покладіть .pdf/.png/.jpg у inputs_docs/, потім:
python add_documents.py                        # усі як repair_invoice, без прив'язки
python add_documents.py --claim CLM-0007       # або одразу до конкретної справи
# ✔ ok: "[add] DOC-L0001 ← ваш_файл.jpg" і pending виріс
```

Файли стають у ту саму чергу, що й демо (`DOC-L…`). Реєстрація моделі не
потребує — **екстракцію** зробить крок 3, коли активний Стек A. Якщо
додали файли ПІСЛЯ перемикання на Стек B — нічого не втрачено: вони
дочекаються наступного інджест-вікна (передеплойте Стек A і повторіть
крок 3). Для вимірюваної точності заповніть `eval/ground_truth_todo.json`
і перенесіть у `ground_truth.json → invoices`.

### Крок 3. Стек A — обробка документів (інджест-вікно)

```bash
cd ../../../inference_jobs/vllm/Qwen2.5-VL-7B-AWQ_banking_and_insurance_docs_min_8gb_vram
cp .env.example .env && python submit.py    # друкує VL_BASE_URL + VL_API_KEY
python predict.py                           # смоук: sample invoice → JSON
# → перенесіть VL_BASE_URL/VL_API_KEY у .env пілота, поверніться сюди:
cd -
python process_documents.py
# ✔ ok: "[ingest] DOC-…(тип) → done", наприкінці by status: extracted≈27+, failed ≤ 3
python run_eval.py --fields                 # точність полів проти ground truth
# → також пише eval/fields_report.html: КАРТИНКА документа поруч із
#   таблицею «еталон vs модель» по кожному полю + причини failed.
#   Відкрийте в браузері — це і є візуальне порівняння (і слайд для демо).
```

Оригінали згенерованих файлів можна погортати і в MinIO-консолі:
<http://localhost:9001> (minioadmin/minioadmin), бакет `bfsi` →
`docs/` (скани) і `policies/` (поліси).

Змінили промпт екстракції чи конфіг моделі? Статуси документів
термінальні, тож чистий перепрогін — це пере-сид:
`python create_demo_data.py && python process_documents.py &&
python run_eval.py --fields` (свої файли з inputs_docs/ після пере-сиду
додайте знову через add_documents.py).

Смоук `predict.py` вважається успішним, якщо: у `models` є `vision`, і
відповідь — JSON (markdown-фенс ```json — норма, воркер його зрізає) з
правильними значеннями полів із зображення. Якщо JSON інколи ламаний —
не страшно: воркер валідує pydantic-ом і робить один retry з текстом
помилки; стабільно ламаний/порожній — дивіться `error` у таблиці
`documents` і `sample_data`-зображення очима.

⚠️ Якщо смоук видає **сміття з перших же токенів** (типу `"vendor_name":":`
і обрив) — це engine-конфіг, не інфраструктура. Задеплойте джобу з
консервативним конфігом (fp8 KV і prefix caching для VL вимкнені — вже
в репо) і дивіться сходинки ескалації в README/serve_module джоби.

### Крок 4. Перемикання стеків (одна карта!)

```bash
# 1) Зупиніть джобу Стека A: дашборд → Inference → job → Stop
# 2) Задеплойте Стек B:
cd ../../../inference_jobs/vllm/Qwen3-8B-AWQ_banking_and_insurance_chat_embed_min_12gb_vram
cp .env.example .env && python submit.py    # друкує CHAT_BASE_URL + CHAT_API_KEY
python predict.py                           # смоук: чат + "embed OK — dim=1024 ✔"
# → CHAT_BASE_URL/CHAT_API_KEY у .env пілота, поверніться:
cd -
python process_documents.py --embed
# ✔ ok: "N embedded, 0 left NULL"
```

### Крок 5. Агент + чат — і питання про ВАШІ документи

```bash
python agent_server.py --selftest           # ✔ ok: "guard ✔ role wall ✔ audit insert ✔"
python agent_server.py &                    # FastAPI на :8000
python chat_cli.py --verbose
```

Що спитати про додані вручну документи (агент дістає їх через
`run_sql`/`get_document` — extracted JSON лежить у БД):

```
you> Що витягнуто з документа DOC-L0001? Покажи всі поля.
you> Покажи всі документи з id на кшталт DOC-L… — суми, вендорів, прив'язку до справ.
you> Які документи лишилися без прив'язки до справи і чому?
you> Рахунок DOC-L0001: чи покривається він полісом своєї справи? Процитуй пункти.
```

Завантаження файлу прямо в чаті теж працює: `/file шлях.jpg` → документ
стає `pending` і обробиться в наступне інджест-вікно (Стек A) — на одній
карті агент чесно про це попередить.

### Крок 6. Оцінка (DoD)

```bash
python run_eval.py                          # 20 питань → eval/report.md
# ✔ ok: pass по питаннях + "hallucinated ids total: 0"
```

**Друга карта в майбутньому:** обидві джоби живуть постійно, перемикання
(крок 4.1) зникає, `/file` у чаті обробляється одразу — код не
змінюється (усі URL з env; `EMBED_BASE_URL` за потреби вкажіть на
окремий деплой).

## Критерії здачі

| Модуль | Критерій |
|---|---|
| М1 | після `create_demo_data.py`: 50 claims, 30 файлів у MinIO, `SELECT count(*) FROM documents WHERE status='pending'` = **30** (20 сканів + 10 полісів) |
| М3 | після `process_documents.py` на Стеку A: всі 30 у extracted/failed, failed ≤ 3 і лише «браковані»; `--embed` на Стеку B → 0 чанків з NULL embedding; `run_eval.py --fields` друкує точність по полях |
| М4 | `agent_server.py --selftest` зелений (guard відбиває UPDATE/другий стейтмент; роль agent_ro не має INSERT/UPDATE окрім audit_log); 5 smoke-питань через chat_cli.py |
| DoD | `run_eval.py`: частка pass по 20 питаннях + **0 галюцинованих id**; звіт у eval/report.md |

## Що навмисно змінено проти вихідної постановки

1. **Docling/Unstructured → regex-чанкер** (`process_documents.py:parse_policy`). Поліси
   генерує наш же `create_demo_data.py` зі стабільною нумерацією `N.` / `N.K.` — зовнішній
   парсер тут не додає нічого, крім залежності. Точка заміни для реальних
   PDF позначена в докстрінгу функції (той самий інтерфейс повернення).
2. **OpenCV → PIL+numpy** для «фото під кутом» (perspective transform + шум) —
   мінус важка залежність.
3. **Три base_url → два деплойменти.** Embed живе в одному деплої з чатом
   (Стек B), `EMBED_BASE_URL` — опційний override на майбутнє.
4. **`policies.vehicle_plate`** додано до схеми — чесний шлях лінкування
   «номер авто на рахунку → поліс → клейм» (у вихідній схемі його не було).
5. **`documents.entity_id = ''`** — сентинел «ще не злінковано» (колонка
   NOT NULL за вихідною схемою).
6. **Pending = 30, не 20** — поліси теж проходять через чергу (їх чанкує
   той самий воркер, модель для цього не потрібна).
7. **Upload у чаті** пише в БД під ingest_rw (вузько скоуплений виняток
   в одному місці agent_server.py) — роль агента лишається read-only.

## Темплейт для інших секторів

Секторна специфіка ізольована в чотирьох місцях — саме їх і міняєте:

| Файл | Що секторне |
|---|---|
| `db/001_core.sql` | доменні таблиці (тут: policies/claims/documents) |
| `create_demo_data.py` | сутності, PDF-шаблони, «червоні» сценарії, ground truth |
| `process_documents.py` | pydantic-схеми екстракції + промпти по doc_type |
| `agent_server.py` | системний промпт (роль/правила), фіксовані red-flag SQL |
| `eval/golden_questions.yaml` | еталонні питання |

Незмінний каркас: compose + ролі БД, цикл воркера (SKIP LOCKED, retry,
confidence), агентський цикл з tools+trace+audit, CLI, метрики eval
(exact-match / цитати / галюциновані id). Приклад іншого сектора на цьому
ж каркасі — construction/BIM
([Qwen3-8B-AWQ_construction_chat…](../../../inference_jobs/vllm/Qwen3-8B-AWQ_construction_chat_min_10gb_vram/README.md)).

## Поза скоупом (свідомо)

Веб-UI, автентифікація бекенда, черги повідомлень, docker для
воркера/бекенда, донавчання моделей, зовнішні реєстри, паралельний інджест.
