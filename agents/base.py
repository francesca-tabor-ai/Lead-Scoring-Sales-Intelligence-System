"""
Base Agent class and Orchestrator
Competitor Intelligence & Market Gap Finder
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import logging
import traceback
import time

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Standardized result envelope returned by every agent."""
    agent_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def __repr__(self):
        status = "✅" if self.success else "❌"
        dur = f" ({self.duration_seconds:.1f}s)" if self.duration_seconds else ""
        return f"{status} {self.agent_name}{dur}"


class Agent(ABC):
    """
    Abstract base class for all pipeline agents.
    Subclasses must implement `run(data)`.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    def run(self, data: Any) -> Any:
        raise NotImplementedError

    def execute(self, data: Any) -> AgentResult:
        """
        Wraps `run()` with timing, structured logging, and error handling.
        """
        started_at = datetime.utcnow()
        self.logger.info(f"[{self.name}] Starting...")
        try:
            result = self.run(data)
            finished_at = datetime.utcnow()
            duration = (finished_at - started_at).total_seconds()
            self.logger.info(f"[{self.name}] Completed in {duration:.2f}s")
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=result,
                started_at=started_at,
                finished_at=finished_at,
            )
        except Exception as e:
            finished_at = datetime.utcnow()
            self.logger.error(f"[{self.name}] Failed: {e}\n{traceback.format_exc()}")
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
                started_at=started_at,
                finished_at=finished_at,
            )

    def __repr__(self):
        return f"<Agent: {self.name}>"


class Orchestrator:
    """
    Sequential multi-agent pipeline orchestrator.
    Each agent's output becomes the next agent's input.
    """

    def __init__(self, agents: List[Agent], stop_on_failure: bool = True):
        self.agents = agents
        self.stop_on_failure = stop_on_failure
        self.logger = logging.getLogger("orchestrator")
        self.run_history: List[AgentResult] = []

    def execute(self, input_data: Any) -> AgentResult:
        """Execute the full pipeline and return the final AgentResult."""
        self.run_history.clear()
        data = input_data
        total_start = time.time()

        self.logger.info(
            f"🚀 Orchestrator starting — {len(self.agents)} agents in pipeline"
        )

        for i, agent in enumerate(self.agents, 1):
            self.logger.info(f"  [{i}/{len(self.agents)}] {agent.name}")
            result = agent.execute(data)
            self.run_history.append(result)

            if not result.success:
                self.logger.error(f"  ❌ '{agent.name}' failed: {result.error}")
                if self.stop_on_failure:
                    return result
            else:
                data = result.data

        elapsed = time.time() - total_start
        successes = sum(1 for r in self.run_history if r.success)
        self.logger.info(
            f"✅ Pipeline complete — {successes}/{len(self.agents)} succeeded "
            f"in {elapsed:.2f}s"
        )

        for result in reversed(self.run_history):
            if result.success:
                return result
        return self.run_history[-1]

    def summary(self) -> str:
        lines = ["Pipeline Summary:"]
        for r in self.run_history:
            lines.append(f"  {r}")
        return "\n".join(lines)
