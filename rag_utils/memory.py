from typing import List, Dict


class SimpleMemory:
    def __init__(self):
        self.chat_history = []

    def append(self, question, answer):
        self.chat_history.append({"question": question, "answer": answer})

    def get_context(self):
        return (
            self.chat_history[-5:] if len(self.chat_history) >= 5 else self.chat_history
        )


class ChatMemory:
    """
    A simple in-memory store for chat history.
    Stores message pairs as a list of dicts: {'role': 'user/assistant', 'content': ...}
    """

    def __init__(self, max_turns: int = 10):
        """
        Initialize the memory with a max number of turns to keep.
        """
        self.chat_history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add_message(self, role: str, content: str):
        """
        Add a message to the history.

        Args:
            role (str): 'user' or 'assistant'
            content (str): Message content
        """
        self.chat_history.append({"role": role, "content": content})
        self.chat_history = self.chat_history[
            -2 * self.max_turns :
        ]  # Each turn has 2 messages

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the chat history.

        Returns:
            List[Dict[str, str]]: Chat history
        """
        return self.chat_history

    def clear(self):
        """Clear the memory."""
        self.chat_history = []


"""
Example Usage :-

memory = ChatMemory(max_turns=5)
memory.add_message("user", "What is RAG?")
memory.add_message("assistant", "RAG stands for Retrieval-Augmented Generation.")
print(memory.get_history())

"""
