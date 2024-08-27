from langchain_core.documents import Document


class RetrieverParser:
    @staticmethod
    def docs_to_context(docs: list[Document]) -> str:
        """
        Parses list of `Document` retrieved from vector store to desired context shape.
        Context shape is currently single string joining all `Document`\'s `answer` attribute
        with separator ' '(whitespace)

        :param docs: List of retrieved `Document`s from vector store
        :return: Single string context used for RAG inference
        """
        answers = [doc.metadata.get("answer", "") for doc in docs]
        return " ".join(answers)
