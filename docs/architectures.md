# Agentic RAG Architectures

This project implements several Retrieval-Augmented Generation (RAG) agents that share common infrastructure (retriever, reranker, `llm_orchestrator`) but add different reasoning loops. The sections below describe how every architecture works, what structured outputs it exchanges, and a small usage example to illustrate the control flow.

---

## Vanilla RAG

**Flow**
1. Encode the user question with the configured embedding model (`generate_query_embedding`).
2. Retrieve top-*k* chunks via vector or hybrid search (`retriever.retrieve`). Optional reranking keeps only the most relevant passages.
3. Call `llm_orchestrator.generate_answer` with the question + selected chunks to get the final response.

**Structured outputs**
- No custom JSON schemas. Everything flows as plain text (question/answer) and Python dicts for chunks.

**Example**
> *Question*: “What tactical innovation is credited to Hungary's Golden Team?”
>
> *Process*: embedding → retrieve 5 chunks → rerank → send chunks to the LLM → LLM cites Puskás's “deep-lying forward” tactic.

---

## Self-Reflective RAG

**Flow**
1. Perform the same retrieval as vanilla (optionally hybrid or active retrieval).
2. Generate an initial answer (`generate_answer`).
3. Invoke the *critique agent* which returns a structured `CritiqueModel` JSON listing missing context, conflicts, hallucinations, etc.
4. Feed the critique + original context to the *refinement agent* (`generate_answer`) which produces an improved answer.
5. Return both the refined answer and telemetry (critique, timings, chunks).

**Structured outputs**
- `CritiqueModel` (Pydantic) – contains lists such as `missing_context`, `conflicts`, `reasoning_gaps`.
- `SelfReflectiveRagOutput` – used when the caller needs a fully structured response (answer text + retrieved chunks + metadata).

**Example**
> *Question*: “At maximal velocity what is the stride length of elite sprinters?”
>
> *Flow*: Retriever gathers biomechanical chunks → initial LLM answer says “about 2.3 m” but forgets to cite context → critique flags missing chunk #D18 → refinement stage rewrites answer citing [chunk_id=c42] and removes hallucinations.

---

## Query Decomposition RAG

**Flow**
1. `QueryPlanner` builds a `QueryDecompositionPlan` with at least two sub-questions (heuristic or LLM-generated).
2. For every plan step:
   - Run a standard mini-RAG (`_execute_subquery_rag`) with optional reranking.
   - Store the `SubQueryAnswer` (question, answer, retrieved chunks, metadata).
3. `AnswerAggregator` receives all sub-answers plus the plan and synthesizes the final reply + integration notes.

**Structured outputs**
- `QueryDecompositionPlan` – ordered steps with `step_id`, `question`, optional heuristics.
- `SubQueryAnswer` – structured record per sub-question.
- `QueryDecompositionOutput` – wraps the plan, sub-answers, final answer, aggregator notes, and timings.

**Example**
> *Question*: “Compare shin angle and hip drive during sprint acceleration.”
>
> *Flow*: Planner splits into `["Explain shin angle mechanics", "Explain hip drive force transfer"]` → each sub-query runs its own retrieval → aggregator stitches both answers and highlights interactions between the factors.

---

## Chain-of-Verification RAG

**Flow**
1. Take any existing answer (usually from vanilla RAG).
2. Use `generate_structured_answer` with the `VerificationPlan` schema to extract 2–N statements requiring verification.
3. For each statement:
   - Retrieve evidence chunks (fresh embedding based on the statement text).
   - Optionally rerank.
   - Ask the verifier LLM for a `StatementAssessment` (status ∈ {verified, contradicted, insufficient}, reasoning).
4. Aggregate results into a new answer that explicitly references verified vs contradicted claims.

**Structured outputs**
- `VerificationPlan` – list of statements to check.
- `VerificationStatement` + `VerificationStatus` enums – persisted in the final `ChainVerificationOutput`.
- `StatementAssessment` – intermediate status + reasoning.

**Example**
> *Question*: “Does Hungary’s Golden Team invent the deep-lying forward?”
>
> *Flow*: Initial answer claims “yes, plus they pioneered zone marking” → statements extracted → evidence for “zone marking” contradicts the answer → final response flags the hallucination while confirming the deep-lying forward fact.

---

## Active Retrieval Agent

**Flow**
1. Start with baseline retrieval and answer.
2. Evaluate sufficiency via a structured `SufficiencyModel` JSON (true/false + reason).
3. If insufficient and iteration limit not reached:
   - Produce a `QueryRewrite` (revised question, reason, `is_final` flag).
   - Re-run retrieval with rewritten query (or custom retriever).
   - Merge newly found chunks with previously collected unique chunks, log stats (`ActiveRetrievalLog`).
4. Repeat until answer deemed sufficient or max iterations reached; return combined context, logs, best answer.

**Structured outputs**
- `QueryRewrite` – revised query, reason, `is_final`.
- `ActiveRetrievalLog` – per-iteration metadata (chunks retrieved, new chunks, reasons).
- Optional `SufficiencyModel` when using default evaluator.

**Example**
> *Question*: “List 3 recovery drills used in elite sprint training.”
>
> *Flow*: Initial retrieval only finds 1 drill → evaluator says insufficient → rewrite query to “recovery drills for sprinters (PNF, mobility, cooldown)” → second iteration finds additional chunks and final answer cites all three drills.

---

## MARAG (Multi-Agent Research & Aggregation)

**Flow**
1. `_planner` generates a `MaragPlan` (role order + researcher iterations).
2. **Researcher** role invokes the Active Retrieval agent to gather exhaustive context (supports query rewriting + multiple passes).
3. **Analyst** deduplicates context and produces structured bullets referencing chunk IDs (`AnalystSummary` composed of `AnalystBullet` items).
4. **Synthesizer** consumes the bullets and creates the final answer, referencing bullet IDs as explicit evidence.
5. Return final answer plus role-specific logs (researcher output, analyst bullets, synthesizer message).

**Structured outputs**
- `MaragPlan`
- `AnalystSummary` / `AnalystBullet` – used to track bullet IDs, chunk references, summaries.

**Example**
> *Question*: “Explain the trade-offs between shin angles and hip drive.”
>
> *Flow*: Researcher performs two iterations of Active Retrieval, logs query rewrites → Analyst emits bullets like `[1] (chunk c21) Shin angle lowers center of mass…` → Synthesizer references `[1][2]` while crafting a cohesive answer.

---

## MADAM-RAG (Multi-Agent Debate and Moderation)

**Flow**
1. Two debater agents independently retrieve context (`_default_retrieve`) and answer the question while citing `chunk_id` values.
2. Moderator reviews both responses; if disagreement remains and `followup_rounds > 0`, it can ask follow-up questions (structured `ModeratorFollowUp` with `ask_followup` flag and question). Debaters respond again using fresh retrievals if needed.
3. Moderator issues a `ModeratorDecision` choosing winner `debater_a`, `debater_b`, or `merged`, with reasoning and a final answer that must cite chunk IDs.
4. All debater responses are stored as `DebaterRecord` → `DebaterResponse` (round_id, answer text, cited chunk IDs) for auditing.

**Structured outputs**
- `DebaterResponse`, `DebaterRecord`
- `ModeratorFollowUp` – decides whether to continue the debate.
- `ModeratorDecision` – winner + final answer text referencing chunk IDs.

**Example**
> *Question*: “Which sport has six players per team?”
>
> *Flow*: Debater A cites volleyball chunks, Debater B disputes with football evidence → Moderator sees conflict, issues a follow-up asking for roster confirmation → final decision declares Debater A the winner with citations `[chunk_id=d_questions_18]`.

---

## Routing Agent

**Flow**
1. Build a routing prompt summarizing available pipelines (`vanilla`, `self_reflective`, `query_decomposition`, `chain_verification`, `active_retrieval`, `marag`, `madam_rag`).
2. Ask the LLM for a `RoutingDecision` JSON (fields: `embedding_profile`, `pipeline`, `iterations`, `followup_rounds`, `reasoning`).
3. Fetch the embedding profile from `config.get_routing_profiles()` (provider, model, description).
4. Execute the selected pipeline via the injected executor map. Some decisions pass parameters (e.g., iterations for chain verification or active retrieval, follow-up rounds for MADAM-RAG).
5. Return a routing payload containing the planner decision, chosen profile, final pipeline output, and metadata for downstream evaluation.

**Structured outputs**
- `RoutingDecision` – ensures the planner always returns valid pipeline names, iteration counts, optional follow-up info.

**Example**
> *Question*: “Summarize Hungary’s Golden Team tactics and validate the claims.”
>
> *Flow*: Routing prompt highlights quality requirements → planner selects `chain_verification` with `iterations=2` and the `balanced_openai` embedding profile → executor runs the chain-of-verification pipeline and returns the verified answer plus the decision metadata.

---

### Tips for Working with These Architectures
- Every agent ultimately relies on the shared `rag_pipeline` retrieval stack; when adding new models or retrievers, ensure the abstractions stay compatible.
- Structured outputs are defined in `src/models/` (Pydantic). They enforce consistent JSON parsing through `llm_orchestrator.generate_structured_answer`.
- The `results/experiment_results.*` files capture per-architecture latency, metric, and routing data so notebook analyses can compare them fairly.

