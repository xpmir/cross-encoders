# Requires transformers>=4.51.0
from sentence_transformers import CrossEncoder


def format_queries(query, instruction=None):
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def format_document(document):
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"


if __name__ == "__main__":
    model = CrossEncoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls")

    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
    ]

    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]

    pairs = [
        [format_queries(query, task), format_document(doc)]
        for query, doc in zip(queries, documents)
    ]
    scores = model.predict(pairs)
    print(scores.tolist())
    # [0.04272603616118431, 0.9991921782493591, 0.40642625093460083, 0.9718492031097412]
