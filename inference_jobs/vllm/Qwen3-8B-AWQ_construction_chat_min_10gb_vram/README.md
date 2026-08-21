# Qwen3-8B (AWQ) — chat baseline + BIM agent testbed (vLLM)

The catalog's cheap-tier chat workhorse (`chat-qwen3-8b`, priority v1,
Apache-2.0): 119 languages (Ukrainian included), hybrid thinking mode, and
reliable **hermes tool calling** — which is why it doubles as the baseline
for the **construction-sector BIM experiment** bundled in this folder
(IFC model → structured digest → questions; then SQL tools).

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 10 → 12 GB | 4 | 24 GB | ~28 GB (image 15 + 1.3×5.5 GB checkpoint) | 12gb |

Suggested GPUs: any 12 GB card (RTX 5070 / 4070 / 3060-12G) runs the full
32k context; a **16 GB card (5070 Ti / 4080)** roughly doubles the KV pool —
same config, more concurrent long-context sessions. Tool calling is **on**
(`enable_auto_tool_choice=True`, `tool_call_parser="hermes"`) — without it,
agent mode (Level B below) half-works at best.

## Folder layout

```
.
├── inference.yaml           # scheduling declaration (12gb tier)
├── scripts/
│   └── serve_module.py      # LLMConfig — 32k ctx, fp8 KV, hermes tools ON
├── sample_data/
│   └── bim_digest.json      # synthetic IfcOpenShell-style extract (45 elements)
├── tools/
│   └── extract_ifc.py       # LOCAL: real .ifc → digest JSON (pip install ifcopenshell)
├── submit.py / predict.py / .env.example
```

`sample_data/` and `tools/` are client-side — only `scripts/` ships to the
cluster.

## Deploy

```bash
cp .env.example .env        # RESON_API_KEY (rsk_…) + S3 keys
python submit.py            # prints RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY
python predict.py "..."     # plain chat smoke test
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## The BIM experiment

**Level A — digest in context (a day of work, no tool calling):**

```bash
python predict.py --bim
python predict.py --bim "Які стіни бетонні? Наведи GUID та об'єми."
```

The digest (an IfcOpenShell extract: guid / class / name / storey /
material / psets incl. BaseQuantities) goes into the context; the system
prompt forbids inventing GUIDs. This tests the core question — does the
model understand construction semantics of the data.

**Level B — SQL agent (the real prototype):**

```bash
python predict.py --bim-tools
python predict.py --bim-tools "Чи є колізії між вентиляцією і несучими конструкціями?"
```

The same digest loads into in-memory SQLite (`elements` / `psets` /
`clashes`) and the model gets `run_sql(query)` + `get_element(guid)`.
Every SQL the model runs is printed — audit whether answers come from data
or imagination. `psets.value` is TEXT: the system prompt tells the model to
`CAST(value AS REAL)` before aggregating.

**Ground truth for `sample_data/bim_digest.json`** (for your ~20-question
eval — exact-answer share + hallucinated-GUID share):

| Question | Answer |
|---|---|
| Doors on Level 2 | **6** (Level 1: 5) |
| Concrete walls | **10** (6 on L1, 4 on L2), net volume **41.92 m³** |
| Total concrete (walls + slabs + stair) | **105.32 m³** |
| Windows total | **10** (4 + 6) |
| Clashes vs load-bearing structures | **2** (duct↔concrete wall, duct↔slab) |

**Real data:** `pip install ifcopenshell`, then

```bash
python tools/extract_ifc.py Duplex_A_20110907.ifc -o sample_data/bim_digest.json
```

Datasets: [Schependomlaan](https://github.com/openBIMstandards/DataSetSchependomlaan)
(real built house + work schedule + site photos — also serves the later
plan-vs-fact stage), Duplex Apartment (`Duplex_A_20110907.ifc`, classic
buildingSMART sample), [Sample-Test-Files](https://github.com/buildingSMART/Sample-Test-Files)
(schema-edge testing). Quantities come from BaseQuantities only if the
designer exported them — otherwise compute via `ifcopenshell.geom` **at
extract time**, never in the agent runtime. Clashes: run `ifcclash`, merge
into the digest's `clashes` list.

**Comparing models:** `predict.py` is a plain OpenAI client — point `.env`
at the [gpt-oss-20b](../gpt-oss-20b_chat_min_14gb_vram/) or
[Qwen3-Coder-30B](../Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit_pipeline_parallel_min_20gb_vram/)
endpoints and rerun the same question set. Note: gpt-oss ships with tool
calling commented out pending parser verification (see its README) —
verify before the Level B comparison, or its results will be unfair.
On 2× 5070 Ti 16 GB: 8B and 20B each fit one card; the 30B needs both
(TP=2 in one box, PP=2 across two machines).

## Does it need fine-tuning?

**Not for Level A/B.** The facts live in the digest / SQL results — a
retrieval-and-tools architecture — and the base model only needs schema
following + SQL, which it has. Fine-tune later (LoRA — see the
[multi-LoRA job](../Qwen2.5-7B_multi_lora_min_10gb_vram/) for the serving
shape) only for surface behavior: Ukrainian construction terminology
(ДСТУ/ДБН phrasing), fixed report formats. Never to "teach it the
building" — that changes per project and belongs in the data layer.

## DWG / 2D plans / paper → IFC (the honest answer)

There is no reliable automatic converter — DWG stores lines, IFC stores
*semantics* (this polyline is a load-bearing wall), and that mapping is
modeling work. The practical route: import the DWG/scan as an underlay in a
BIM tool (**Bonsai (Blender)** — free, Revit/Archicad/FreeCAD-BIM likewise)
and re-model over it, then export IFC. Paper plans: scan → raster underlay →
same. View/verify results with **IfcOpenShell** (data), **Bonsai** (3D
editing), **xeokit** (web viewer). Budget re-modeling as the paid setup
service, not a batch script.
