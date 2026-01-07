from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """
    Central state object passed through the LangGraph workflow.
    common state of all the agents
    """

 
    input: str
    context: Dict[str, Any] = field(default_factory=dict)
    proposed_action: Optional[str] = None
    executed_action: Optional[str] = None
    observation: Optional[str] = None
    logs: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Appening a message to the execution log."""
        self.logs.append(message)