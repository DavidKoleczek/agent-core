import asyncio
import os
from pathlib import Path
from typing import Literal, cast

from anthropic import AsyncAnthropic
import click
from google import genai
from interop_router.router import Router
from interop_router.types import SupportedModel
from openai import AsyncOpenAI

from agent_core._types import AgentConfig, SubagentConfig, TurnConfig, UserMessageEvent
from agent_core.agent import Agent
from agent_core.tools.presets import permissive_tools, standard_tools


async def run_agent(
    prompt: str,
    working_dir: Path,
    mode: Literal["standard", "permissive"],
    model: str,
    model_friendly_name: str,
    model_knowledge_cutoff: str,
    timezone: str,
) -> None:
    router = Router()
    router.register("openai", AsyncOpenAI())
    router.register("gemini", genai.Client(api_key=os.getenv("GEMINI_API_KEY")))
    router.register("anthropic", AsyncAnthropic())

    agent_config = AgentConfig(working_dir=working_dir, max_subagent_depth=1)

    if mode == "permissive":
        tools = permissive_tools(working_dir=working_dir)
    else:
        tools = standard_tools(working_dir=working_dir)

    # Base turn config for subagents (without their own subagents)
    base_turn_config = TurnConfig(
        model=cast(SupportedModel, model),
        model_friendly_name=model_friendly_name,
        model_knowledge_cutoff=model_knowledge_cutoff,
        timezone=timezone,
        tools=tools,
        subagents=[],
    )

    # Main agent turn config with subagents enabled
    turn_config = TurnConfig(
        model=cast(SupportedModel, model),
        model_friendly_name=model_friendly_name,
        model_knowledge_cutoff=model_knowledge_cutoff,
        timezone=timezone,
        tools=tools,
        subagents=[
            SubagentConfig(
                name="general-purpose",
                description="General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks.",
                turn_config=base_turn_config,
            ),
        ],
    )

    agent = Agent(agent_config, router)
    user_event = UserMessageEvent(message=prompt)

    async for event in agent.turn(user_event, turn_config):
        print("EVENT:", str(event)[0:340], sep="", end="\n\n")


@click.command()
@click.option(
    "--prompt",
    default="Use a general sub-agent task to quickly explore this codebase. Only read the 3 most important files in parallel. Don't use todos for this",
    help="The prompt to send to the agent.",
)
@click.option(
    "--working-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("/home/david/repos/agent-core"),
    help="The working directory for the agent.",
)
@click.option(
    "--mode",
    type=click.Choice(["standard", "permissive"]),
    default="permissive",
    help="Tool permission mode.",
)
@click.option("--model", default="gpt-5.1-codex-max", help="Model ID for API calls.")
@click.option("--model-friendly-name", default="gpt-5.1-codex-max", help="Human-readable model name.")
@click.option("--model-knowledge-cutoff", default="Sep 30, 2024", help="Model knowledge cutoff date.")
@click.option("--timezone", default="America/New_York", help="IANA timezone name.")
def main(
    prompt: str,
    working_dir: Path,
    mode: Literal["standard", "permissive"],
    model: str,
    model_friendly_name: str,
    model_knowledge_cutoff: str,
    timezone: str,
) -> None:
    """Run an AI agent with the given prompt."""
    asyncio.run(
        run_agent(
            prompt=prompt,
            working_dir=working_dir,
            mode=mode,
            model=model,
            model_friendly_name=model_friendly_name,
            model_knowledge_cutoff=model_knowledge_cutoff,
            timezone=timezone,
        )
    )


if __name__ == "__main__":
    main()
