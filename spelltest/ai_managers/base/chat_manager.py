from abc import ABC, abstractmethod
from typing import List
from ...entities.managers import ConversationState, MessageType, Message


class ChatManagerBase(ABC):
    def __init__(self, role: MessageType, opposite_role: MessageType, *args, **kwargs):
        self.your_role = role
        self.opposite_role = opposite_role
        self.chat_history: List[Message] = []
        self.state = ConversationState.CREATED
        if hasattr(self, "target_prompt") and not hasattr(self, "prompt_version_id"):
            self.prompt_version_id = self.target_prompt.promptelligence_params.db_version_id
        if role == "Human":
            self._post_init_check()

    def _post_init_check(self):
        if not hasattr(self, "metrics"):
            raise TypeError(f"Instances of {self.__class__.__name__} must have a `metrics` attribute.")
        if not hasattr(self, "prompt_version_id"):
            raise TypeError(f"Instances of {self.__class__.__name__} must have a `prompt_version_id` attribute.")

    @abstractmethod
    def initialize_conversation(self):
        """
        Initialize conversation if required
        """
        pass

    @abstractmethod
    def next_message(self, message: Message) -> Message:
        """Process the next message"""
        pass

    def finish(self) -> bool:
        """
        Finish the conversation here
        :return: bool
        """
        self.state = ConversationState.FINISHED

    def conversation_state(self):
        return self.state