from prometheus_client import Counter, Histogram

QUESTIONS = Counter(
    "tax_assistant_questions",
    "Questions processed by the RAG pipeline, by outcome.",
    labelnames=["answered"],
)

LLM_DURATION = Histogram(
    "tax_assistant_llm_duration_seconds",
    "Duration of the LLM call in the answer node.",
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60],
)
