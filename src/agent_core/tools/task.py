from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING
import uuid

from interop_router.router import Router
from liquid import render
from openai.types.responses.function_tool_param import FunctionToolParam

from agent_core._types import AgentConfig, AgentEvent, AgentMessageEvent, AgentTurnEnd, SubagentConfig, UserMessageEvent
from agent_core.tools._utils import ConstraintPolicy

if TYPE_CHECKING:
    from agent_core.agent import Agent

TOOL_NAME = "task"

TOOL_DESCRIPTION_TEMPLATE = """\
Launch a new agent to handle complex, multi-step tasks autonomously. Do not use this tool if you can accomplish the task directly.

The Task tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it. \
When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

Available agent types:
{% for subagent in subagents %}- {{ subagent.name }}: {{ subagent.description }}
{% endfor %}
When NOT to use the Task tool:
- If you want to read a specific file path, use the Read or Glob tool instead of the Task tool, to find the match more quickly
- If you are searching for a specific class definition like "class Foo", use the Glob tool instead, to find the match more quickly
- If you are searching for code within a specific file or set of 2-3 files, use the Read tool instead of the Task tool, to find the match more quickly
- Do not use the Task for trivial tasks or general responses.
- Other tasks that are not related to the agent descriptions above

Usage notes:
- Always include a short description (3-5 words) summarizing what the agent will do
- Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
- When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
- Provide clear, detailed prompts so the agent can work autonomously and return exactly the information you need.
- Agents with "access to current context" can see the full conversation history before the tool call. When using these agents, you can write concise prompts that reference earlier context (e.g., "investigate the error discussed above") instead of repeating information. The agent will receive all prior messages and understand the context.
- The agent's outputs should generally be trusted
- Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
- If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
- If the user specifies that they want you to run agents "in parallel", you MUST send a single message with multiple Task tool use content blocks. For example, if you need to launch both a code-reviewer agent and a test-runner agent in parallel, send a single message with both tool calls.

Example usage:

<example_agent_descriptions>
"code-reviewer": use this agent after you are done writing a significant piece of code
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
</example_agent_description>

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
assistant: First let me use the Write tool to write a function that checks if a number is prime
assistant: I'm going to use the Write tool to write the following code:
<code>
function isPrime(n) {
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {
    if (n % i === 0) return false
  }
  return true
}
</code>
<commentary>
Since a significant piece of code was written and the task was completed, now use the code-reviewer agent to review the code
</commentary>
assistant: Now let me use the code-reviewer agent to review the code
assistant: Uses the Task tool to launch the code-reviewer agent
</example>

<example>
user: "Hello"
<commentary>
Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
</commentary>
assistant: "I'm going to use the Task tool to launch the greeting-responder agent"
</example>"""


def build_task_tool_definition(subagents: Sequence[SubagentConfig]) -> FunctionToolParam:
    """Build the task tool definition with available agent types."""
    description = render(TOOL_DESCRIPTION_TEMPLATE, subagents=subagents)
    subagent_names = [s.name for s in subagents]

    return {
        "type": "function",
        "name": TOOL_NAME,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "A short (3-5 word) description of the task"},
                "prompt": {"type": "string", "description": "The task for the agent to perform"},
                "subagent_type": {
                    "type": "string",
                    "description": "The type of specialized agent to use for this task",
                    "enum": subagent_names,
                },
            },
            "required": ["description", "prompt", "subagent_type"],
            "additionalProperties": False,
        },
        "strict": True,
    }


class TaskTool:
    TOOL_NAME = TOOL_NAME

    def __init__(
        self,
        agent_class: "type[Agent]",
        agent_config: AgentConfig,
        router: Router,
        subagents: Sequence[SubagentConfig],
        parent_agent: "Agent",
    ) -> None:
        self._agent_class = agent_class
        self._agent_config = agent_config
        self._router = router
        self._subagents = subagents
        self._subagents_by_name = {s.name: s for s in subagents}
        self._parent_agent = parent_agent

        # Build tool definition dynamically based on available subagents
        tool_definition = build_task_tool_definition(subagents)
        self.TOOLS: dict[str, FunctionToolParam] = {TOOL_NAME: tool_definition}

    async def execute(self, **arguments: object) -> AsyncGenerator[AgentEvent, None]:
        """Execute a task by spawning a subagent.

        Args:
            **arguments: Tool arguments containing:
                - description: Short (3-5 word) description of the task.
                - prompt: The task for the agent to perform.
                - subagent_type: The type of specialized agent to use.

        Returns:
            The final message from the subagent, or an error string.
        """
        prompt = str(arguments.get("prompt", ""))
        subagent_type = str(arguments.get("subagent_type", ""))

        if subagent_type not in self._subagents_by_name:
            available = list(self._subagents_by_name.keys())
            yield AgentMessageEvent(
                message=f"Unsupported subagent_type: {subagent_type}. Available types: {available}",
                agent_id=self._parent_agent.agent_id,
            )
            yield AgentTurnEnd(reason="completed", agent_id=self._parent_agent.agent_id)
            return

        # Derive subagent chat file from parent's chat file
        parent_chat_file = self._agent_config.chat_file
        if parent_chat_file is None:
            yield AgentMessageEvent(
                message="Parent agent has no chat file configured.", agent_id=self._parent_agent.agent_id
            )
            yield AgentTurnEnd(reason="completed", agent_id=self._parent_agent.agent_id)
            return

        # Get the subagent's TurnConfig
        subagent_config_entry = self._subagents_by_name[subagent_type]
        subagent_turn_config = subagent_config_entry.turn_config

        # Calculate depth and determine if subagent can spawn its own subagents
        current_depth = len(self._parent_agent._ancestors) + 1
        max_depth = self._agent_config.max_subagent_depth
        can_spawn_subagents = current_depth < max_depth
        subagent_ancestors = (*self._parent_agent._ancestors, self._parent_agent)

        # If subagent cannot spawn its own subagents, clear the subagents list
        if not can_spawn_subagents:
            subagent_turn_config = subagent_turn_config.model_copy(update={"subagents": []})

        # Build the agent_name chain, excluding "main" from the chain
        parent_agent_name = self._parent_agent._agent_name
        subagent_name = f"{parent_agent_name} -> {subagent_type}" if parent_agent_name != "main" else subagent_type

        short_uuid = str(uuid.uuid4())[:8]
        subagent_chat_file = parent_chat_file.with_stem(f"{parent_chat_file.stem}_{short_uuid}")

        try:
            subagent_agent_config = AgentConfig(
                working_dir=self._agent_config.working_dir,
                chat_file=subagent_chat_file,
                max_subagent_depth=max_depth,
            )

            subagent = self._agent_class(
                config=subagent_agent_config,
                router=self._router,
                ancestors=subagent_ancestors,
                agent_name=subagent_name,
            )

            user_event = UserMessageEvent(message=prompt)
            async for event in subagent.turn(user_event, subagent_turn_config):
                yield event

        except Exception as e:
            yield AgentMessageEvent(message=f"Subagent execution failed: {e}", agent_id=self._parent_agent.agent_id)
            yield AgentTurnEnd(reason="completed", agent_id=self._parent_agent.agent_id)

    def check_constraint(self, **arguments: object) -> ConstraintPolicy:
        return ConstraintPolicy.ALLOW
