from functools import partial
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langgraph.graph.state import CompiledStateGraph, StateGraph

from core.config import Settings
from rag.language import LANGUAGE_NAMES, Language, detect_language

NO_ANSWER: dict[Language, str] = {
    "hu": "A NAV füzetek nem tárgyalják ezt a kérdést.",
    "en": "The booklets don't cover this question.",
}
PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You answer questions using only the NAV information booklets below. Use ONLY the following tax "+
    "booklets to answer the question. Cite the title and page of the "+
    "source for each claim. You can only give information, you can not give any advice, or calculations beyond the "+
    "passages. If the answer is not in the booklets, reply exactly: {no_answer}"),
    ("user", "Passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer in {language_name}."),
])

class QAState(TypedDict, total=False):
    question: str
    language: Language
    top_k: int
    documents: list[tuple[Document, float]]
    answer: str
    answered: bool


def build_graph(vector_store: VectorStore, chat_model: BaseChatModel, settings: Settings) -> CompiledStateGraph:
    graph = StateGraph(QAState)
    graph.add_node("detect_language", detect_language_node)
    graph.add_node("retrieve", partial(retrieve, vector_store=vector_store, min_similarity=settings.min_similarity))
    graph.add_node("no_answer", no_answer)
    graph.add_node("answer", partial(answer, chat_model=chat_model))
    graph.set_entry_point("detect_language")
    graph.add_edge("detect_language", "retrieve")
    graph.add_conditional_edges("retrieve", route, ["answer", "no_answer"])
    graph.set_finish_point("no_answer")
    graph.set_finish_point("answer")
    return graph.compile()

async def route(state: QAState) -> str:
    return "answer" if state["documents"] else "no_answer"


async def detect_language_node(state: QAState) -> dict:
    return {"language": detect_language(state["question"])}


async def retrieve(state: QAState, vector_store: VectorStore, min_similarity: float) -> dict:
    hits = await vector_store.asimilarity_search_with_score(state["question"], k=state["top_k"])
    return {"documents": [(doc, score) for doc, score in hits if score >= min_similarity]}


async def no_answer(state: QAState) -> dict:
    return {"answer": NO_ANSWER[state["language"]], "answered": False}


def format_context(documents: list[tuple[Document, float]]) -> str:
    blocks = []
    for i, (doc, _score) in enumerate(documents, start=1):
        header = f"[{i}] {doc.metadata.get('title', doc.metadata.get('source'))}, {doc.metadata.get('page')}. oldal"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


async def answer(state: QAState, chat_model: BaseChatModel) -> dict:
    context = format_context(state["documents"])
    language = state["language"]
    messages = PROMPT.format_messages(
        context=context,
        question=state["question"],
        language_name=LANGUAGE_NAMES[language],
        no_answer=NO_ANSWER[language],
    )
    response = await chat_model.ainvoke(messages)
    answered = not any(sentence.rstrip(".") in response.content for sentence in NO_ANSWER.values())
    return {"answer": response.content, "answered": answered}
