"""Figure 3 (English) system + user prompt templates, verbatim from the RGB paper."""

SYSTEM_INSTRUCTION = (
    "You are an accurate and reliable AI assistant that can answer questions with "
    "the help of external documents. Please note that external documents may "
    "contain noisy or factually incorrect information. If the information in the "
    "document contains the correct answer, you will give an accurate answer. If "
    "the information in the document does not contain the answer, you will "
    "generate 'I can not answer the question because of the insufficient "
    "information in documents.' If there are inconsistencies with the facts in "
    "some of the documents, please generate the response 'There are factual "
    "errors in the provided documents.' and provide the correct answer."
)

USER_TEMPLATE = "Document:\n{docs}\n\nQuestion:\n{query}"


def format_docs(docs: list[str]) -> str:
    """Numbered concatenation of the passages shown to the model."""
    return "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))


def build_prompt(query: str, docs: list[str]) -> tuple[str, str]:
    """Returns (system_instruction, user_prompt) exactly per Figure 3."""
    return SYSTEM_INSTRUCTION, USER_TEMPLATE.format(docs=format_docs(docs), query=query)


# --- Prompts for the mitigation techniques (section 4 of README) -----------------

VERIFY_THEN_ANSWER_SYSTEM = (
    SYSTEM_INSTRUCTION
    + " Before answering, first privately check whether ANY document actually "
    "supports an answer; if none do, you must output only the refusal sentence "
    "above and nothing else."
)

DECOMPOSE_SYSTEM = (
    "You are an accurate and reliable AI assistant. Split the user's question "
    "into the minimal list of independent sub-questions needed to answer it "
    "fully. Return ONLY the sub-questions, one per line, no numbering, no extra text."
)

MERGE_SYSTEM = (
    "You are an accurate and reliable AI assistant. You are given a question and "
    "the answers to its sub-questions. Combine them into a single, complete final "
    "answer that addresses every part of the original question. Do not drop any "
    "part of the answer."
)

PARAMETRIC_ONLY_SYSTEM = (
    "You are a knowledgeable AI assistant. Answer the question using only your "
    "own internal knowledge, with no external documents. Be concise."
)

CROSSCHECK_SYSTEM = (
    "You are an accurate and reliable AI assistant that fact-checks retrieved "
    "documents against a model's own internal knowledge. You will be given a "
    "question, the model's own answer (no documents), and external documents. "
    "If the documents agree with the model's own answer, confirm it. If the "
    "documents contradict the model's own knowledge, generate 'There are factual "
    "errors in the provided documents.' and then give the correct answer, trusting "
    "your own knowledge unless the documents provide strong independent evidence "
    "otherwise."
)

CROSSCHECK_USER_TEMPLATE = (
    "Question:\n{query}\n\nModel's own answer (no documents):\n{own_answer}\n\n"
    "Document:\n{docs}"
)
