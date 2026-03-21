"""Middleware for writing per-thread agent run history to JSONL files."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from deerflow.agents.thread_state import ThreadDataState
from deerflow.config.paths import get_paths


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_log_dir() -> Path:
    return get_paths().base_dir / "channels" / "run_events"


class RunHistoryMiddlewareState(AgentState):
    """Compatible with ThreadState fields used by the middleware."""

    thread_data: ThreadDataState | None
    title: str | None
    artifacts: list[str] | None
    todos: list | None


class RunHistoryMiddleware(AgentMiddleware[RunHistoryMiddlewareState]):
    """Write per-run agent history snapshots to JSONL for local debugging."""

    state_schema = RunHistoryMiddlewareState

    def __init__(self) -> None:
        super().__init__()
        self._enabled = _env_truthy("DEERFLOW_RUN_EVENT_LOG_ENABLED", default=False)
        self._log_dir = Path(os.getenv("DEERFLOW_RUN_EVENT_LOG_DIR", str(_default_log_dir())))
        if self._enabled:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _serialize_message(msg: Any) -> dict[str, Any]:
        if isinstance(msg, AIMessage):
            data: dict[str, Any] = {
                "type": "ai",
                "content": msg.content,
                "id": getattr(msg, "id", None),
            }
            if getattr(msg, "tool_calls", None):
                data["tool_calls"] = [
                    {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
                    for tc in msg.tool_calls
                ]
            return data
        if isinstance(msg, ToolMessage):
            return {
                "type": "tool",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                "name": getattr(msg, "name", None),
                "tool_call_id": getattr(msg, "tool_call_id", None),
                "id": getattr(msg, "id", None),
                "status": getattr(msg, "status", None),
            }
        if isinstance(msg, HumanMessage):
            return {
                "type": "human",
                "content": msg.content,
                "id": getattr(msg, "id", None),
                "additional_kwargs": getattr(msg, "additional_kwargs", {}),
            }
        if isinstance(msg, SystemMessage):
            return {"type": "system", "content": msg.content, "id": getattr(msg, "id", None)}
        return {"type": getattr(msg, "type", "unknown"), "content": str(msg), "id": getattr(msg, "id", None)}

    def _append_record(self, thread_id: str, record: dict[str, Any]) -> None:
        safe_thread_id = "".join(ch for ch in thread_id if ch.isalnum() or ch in {"-", "_"}) or "unknown"
        output_path = self._log_dir / f"{safe_thread_id}.jsonl"
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def after_agent(self, state: RunHistoryMiddlewareState, runtime: Runtime) -> dict | None:
        if not self._enabled:
            return None

        thread_id = runtime.context.get("thread_id")
        if not thread_id:
            return None

        messages = state.get("messages", [])
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "lead_agent.run.complete",
            "thread_id": thread_id,
            "agent_name": runtime.context.get("agent_name"),
            "title": state.get("title"),
            "artifacts": state.get("artifacts", []),
            "todos": state.get("todos"),
            "message_count": len(messages),
            "messages": [self._serialize_message(msg) for msg in messages],
        }

        try:
            self._append_record(thread_id, record)
        except Exception:
            # Avoid affecting the agent run when debug logging fails.
            return None

        return None