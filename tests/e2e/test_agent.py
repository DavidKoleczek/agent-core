import os
from pathlib import Path

from anthropic import AsyncAnthropic
from google import genai
from interop_router.router import Router
from openai import AsyncOpenAI

from agent_core._types import AgentConfig, AgentTurnEnd, SubagentConfig, TurnConfig, UserMessageEvent
from agent_core.agent import Agent
from agent_core.tools.presets import standard_tools

CALCULATOR_PY = """\
def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b
"""

GREETINGS_PY = """\
def hello(name: str) -> str:
    return f"Hello, {name}!"


def goodbye(name: str) -> str:
    return f"Goodbye, {name}!"
"""

DATA_PROCESSOR_PY = """\
def uppercase_all(items: list[str]) -> list[str]:
    return [item.upper() for item in items]


def filter_long(items: list[str], min_length: int = 5) -> list[str]:
    return [item for item in items if len(item) >= min_length]
"""


def _create_sample_files(directory: Path) -> None:
    (directory / "calculator.py").write_text(CALCULATOR_PY)
    (directory / "greetings.py").write_text(GREETINGS_PY)
    (directory / "data_processor.py").write_text(DATA_PROCESSOR_PY)


def _create_router() -> Router:
    router = Router()
    router.register("openai", AsyncOpenAI())
    router.register("gemini", genai.Client(api_key=os.getenv("GEMINI_API_KEY")))
    router.register("anthropic", AsyncAnthropic())
    return router


async def test_simple_subagent(tmp_path: Path) -> None:
    _create_sample_files(tmp_path)

    router = _create_router()
    agent_config = AgentConfig(working_dir=tmp_path, max_subagent_depth=1)
    tools = standard_tools(working_dir=tmp_path)

    # Base config for subagents
    base_turn_config = TurnConfig(
        model="gpt-5.1-codex-max",
        model_friendly_name="gpt-5.1-codex-max",
        model_knowledge_cutoff="Sep 30, 2024",
        timezone="America/New_York",
        tools=tools,
        subagents=[],
    )

    turn_config = TurnConfig(
        model="gpt-5.1-codex-max",
        model_friendly_name="gpt-5.1-codex-max",
        model_knowledge_cutoff="Sep 30, 2024",
        timezone="America/New_York",
        tools=tools,
        subagents=[
            SubagentConfig(
                name="general-purpose",
                description="General-purpose agent for exploration",
                turn_config=base_turn_config,
            ),
        ],
    )
    agent = Agent(agent_config, router)
    user_event = UserMessageEvent(
        message="Use a subagent to explore each of the files in this directory and summarize to me what it's about"
    )

    events = []
    async for event in agent.turn(user_event, turn_config):
        events.append(event)

    turn_end_events = [e for e in events if isinstance(e, AgentTurnEnd)]
    assert len(turn_end_events) > 0, "Agent should produce a turn end event"


async def test_agent_web_search_and_write(tmp_path: Path) -> None:
    """
    Test that the agent can search the web and write results to a file.
    TODO: There may be a bug where built-in tools require approval.
    """
    router = _create_router()
    agent_config = AgentConfig(working_dir=tmp_path, max_subagent_depth=1)
    tools = standard_tools(working_dir=tmp_path)
    turn_config = TurnConfig(
        model="gpt-5.2",
        model_friendly_name="gpt-5.2",
        model_knowledge_cutoff="Sep 30, 2024",
        timezone="America/New_York",
        tools=tools,
    )
    agent = Agent(agent_config, router)
    user_event = UserMessageEvent(
        message="Look up the latest news as of today and write a short summary to a new file called news_summary.md"
    )

    events = []
    async for event in agent.turn(user_event, turn_config):
        events.append(event)

    turn_end_events = [e for e in events if isinstance(e, AgentTurnEnd)]
    assert len(turn_end_events) > 0, "Agent should produce a turn end event"

    summary_file = tmp_path / "news_summary.md"
    assert summary_file.exists(), "Agent should create news_summary.md"
    assert len(summary_file.read_text()) > 0, "Summary file should have content"
