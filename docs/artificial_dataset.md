# Artificial Knowledge Base and Question Design

This repository does not rely on a publicly available corpus. Instead, we crafted a lightweight “micro world” that mixes futuristic policy narratives with contemporary sports material so that every agentic RAG architecture can be stress-tested. This note explains what each artificial file contains, the design intent behind it, and how you can reuse or extend the material.

## Files that drive the experiments

| File | Purpose | Notes |
| --- | --- | --- |
| `data/future_poland/questions.csv` | Canonical evaluation set used by the notebooks. | Contains hand-written prompts that target different reasoning modes (multi-hop comparisons, fact-checking, chain-of-thought, routing stress-tests). Some questions explicitly pit two sports against each other (“football or volleyball?”, “which drill is better…”) so hybrid retrievers and rerankers are exercised. |
| `data/future_poland/d_questions.json` | JSON flavor of the same evaluation set. | Keeps extra metadata fields (difficulty, category) to enable routing or curriculum-style experiments. |
| `data/future_poland/raw/*.md` | Knowledge base. Split into two families described below. | Files were written to include subtle contradictions, cross-references, and values designed for factual grounding tests. |
| `data/future_poland/processed/chunks.json` | Pre-split chunks used by the retriever. | Generated from the markdown sources. Each chunk preserves the `chunk_id`, which allows MADAM-RAG and Chain-of-Verification to cite evidence. |

## Futuristic policy series `D01.md` – `D20.md`

These twenty markdown files are narrative briefs that chronicle “Poland’s orbital century” from 2025 to 2125. Every document focuses on a theme (digital governance, modular nuclear energy, orbital manufacturing nodes, cognitive democracy, etc.) and is intentionally dense with:

- **Canonical facts**: milestone identifiers such as “F1” (first nuclear plant) or “six orbital nodes by 2125” are repeated so multi-hop queries can track them through time.
- **Divergent footnotes**: many sections mention slight discrepancies (e.g., “some sources say the plant finished in 2036”) to test verification agents.
- **Cross-file dependencies**: D01–D05 set up the early groundwork, D06–D13 describe social/energy pivots, and D14–D20 explore orbital logistics, AI governance, and Baltic security in 2100+. Answering long-form policy questions usually requires stitching data from several files.

**Example use case**
- Question: “Why did Poland settle on six orbital manufacturing nodes by 2125 and what dual cost signals guided logistics?”  
- Required hops: D18 (logistics signals) + earlier files describing nuclear investments (e.g., D03/D05) + governance references (D14+). Chain-of-Verification has to spot the dual estimates (9,800 USD vs 7,400 USD) while self-reflective agents refine the narrative.

## Sports & performance knowledge (`football_rules.md`, `football_tactics.md`, `volleyball_rules.md`, `sprinting_mechanics.md`, `training_periodization.md`)

This second family supports the short, factoid-style questions. Each file intentionally overlaps in scope so that hybrid retrieval, reranking, and routing agents must decide which modality (lexical vs dense) works best.

- **Rules files** stress precision: they include player counts, substitution rules, scoring, and edge cases. Several statements contradict pop-culture myths (“football has 11 players, volleyball 6”) to test hallucination resistance.
- **Tactics & mechanics** mix qualitative descriptions with numeric cues (stride length ranges, acceleration angles). Questions often require comparing two chunks (multi-hop) or distinguishing between similar-looking drills.
- **Training_periodization** introduces sequences such as “base → load → taper” so query-decomposition agents can split the reasoning (“describe load week intensity” vs “explain taper benefits”).

**Example use case**
- Question: “Which sport has teams of six players: football or volleyball?”  
- Files involved: `football_rules.md` (explicitly 11 players), `volleyball_rules.md` (6 players). Routing or MADAM-RAG agents must retrieve both and reason over the contrast, while Chain-of-Verification can double-check the exact line.

## How these files stress the agents

1. **Multi-hop reasoning** – Many questions cannot be answered from a single chunk. For instance, orbital logistics questions need timeline + cost files, while training questions need both mechanics and periodization files.
2. **Contradictions & ambiguity** – Intentional discrepancies (e.g., F1 = 2035 vs 2036) force verification agents to note uncertainty rather than hallucinate a single fact.
3. **Routing diversity** – The dataset covers futuristic policy, sports rules, and biomechanics. The routing prompt must decide which pipeline to call (self-reflective for policy, vanilla for quick factoids, MADAM-RAG for debates).
4. **Structured outputs** – Several agents rely on Pydantic schemas (critique, bullets, verification statements, moderator decisions). The chunk IDs in `chunks.json` ensure those schemas can cite specific evidence.

## Extending the artificial corpus

- Add new markdown files under `data/future_poland/raw/` following the same pattern: dense content, embedded canonical facts, and subtle “what-if” deviations.
- Update `data/future_poland/questions.csv` with new categories (e.g., finance, medicine). Keep at least a few multi-hop prompts so Query Decomposition and Chain-of-Verification have room to shine.
- Regenerate processed chunks (`python scripts/build_chunks.py`, if available) so retrieval stays in sync.

By documenting the intent of each artificial file, we can reproduce experiments deterministically, explain agent behavior during demos, and onboard collaborators quickly. Feel free to link this note from your presentations or README if you need to justify why the evaluation questions look “too tailored” — that tailoring is deliberate to validate the agentic architectures.
