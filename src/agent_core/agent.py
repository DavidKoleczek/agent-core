from collections.abc import AsyncGenerator, Sequence
import copy
from datetime import datetime
import inspect
import json
from pathlib import Path
import re
import tempfile
from typing import Any
import uuid

from interop_router.router import Router
from interop_router.types import ChatMessage
from liquid import render
from openai.types.responses import EasyInputMessageParam, WebSearchToolParam
from openai.types.responses.response_input_item_param import FunctionCallOutput

from agent_core._types import (
    AgentChatHistory,
    AgentConfig,
    AgentEvent,
    AgentMessageEvent,
    AgentReasoningEvent,
    AgentToolCallEvent,
    AgentToolOutputEvent,
    AgentTurnEnd,
    TurnConfig,
    UserEvent,
    UserMessageEvent,
    UserToolCallPermissionEvent,
)
from agent_core.hooks import git, system_info
from agent_core.tools._protocol import Tool
from agent_core.tools._utils import ConstraintPolicy
from agent_core.tools.task import TaskTool

POLICY_TO_STATUS: dict[ConstraintPolicy, str] = {
    ConstraintPolicy.ALLOW: "approved",
    ConstraintPolicy.ASK: "pending",
    ConstraintPolicy.DENY: "denied",
}

PERMISSION_TO_STATUS: dict[str, str] = {
    "accept": "approved",
    "deny": "denied",
}


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        router: Router,
        ancestors: Sequence["Agent"] = (),
        agent_name: str = "main",
    ):
        """Initialize the Agent.

        Args:
            config: Agent configuration with working directory and chat file settings.
            router: Router for API communication.
            ancestors: Chain of parent Agent instances for tracking recursion depth.
            agent_name: Friendly name for this agent, used in events. For subagents, this
                is the chain of subagent types (e.g., "general-purpose -> general-purpose").
        """
        self._agent_name = agent_name
        self.config = config
        self.router = router
        self._ancestors = ancestors

        self._chat_file = self._resolve_chat_file()
        self.config = config.model_copy(update={"chat_file": self._chat_file})
        self.history: list[AgentChatHistory] = self._load_history()

    @property
    def agent_id(self) -> str:
        """Identifier for this agent, derived from chat file stem."""
        return self._chat_file.stem

    async def turn(self, user_event: UserEvent, turn_config: TurnConfig) -> AsyncGenerator[AgentEvent, None]:
        # Create task tool for this turn (if subagents configured)
        task_tool = self._create_task_tool(turn_config)

        # Build lookup dict: tool_name -> tool instance
        tool_by_name: dict[str, Tool] = {}
        for tool in turn_config.tools:
            for name in tool.TOOLS:
                tool_by_name[name] = tool

        if isinstance(user_event, UserToolCallPermissionEvent):
            self._update_tool_permission(user_event.call_id, user_event.permission)
            if user_event.feedback:
                self.history.append(
                    AgentChatHistory(message=EasyInputMessageParam(role="user", content=user_event.feedback))
                )
            await self._save_history()
        elif isinstance(user_event, UserMessageEvent):
            self.history.append(
                AgentChatHistory(message=EasyInputMessageParam(role="user", content=user_event.message))
            )
            await self._save_history()

        # Execute any tools that are now approved (or generate denied outputs)
        async for event in self._execute_tools(task_tool, tool_by_name):
            yield event

        # If any tool calls still pending, wait for decisions
        if self._has_pending_tool_calls():
            yield AgentTurnEnd(reason="tools_need_decision", agent_id=self.agent_id, agent_name=self._agent_name)
            return

        while True:
            # Render system prompt with context
            working_dir = str(self.config.working_dir)
            system_prompt = render(
                turn_config.system_prompt_template,
                working_directory=working_dir,
                is_git_repo=git.is_git_repo(working_dir),
                platform=system_info.platform(),
                os_version=system_info.os_version(),
                current_date=system_info.todays_date(turn_config.timezone),
                model_friendly_name=turn_config.model_friendly_name,
                model_id=turn_config.model,
                knowledge_cutoff=turn_config.model_knowledge_cutoff,
                current_branch=git.current_branch(working_dir) or "N/A",
                main_branch=git.main_branch(working_dir) or "N/A",
                git_status=git.git_status(working_dir) or "N/A",
                recent_commits=git.recent_commits(working_dir) or "N/A",
            )

            # Create a copy of history to modify for the model call
            history_copy = copy.deepcopy(self.history)
            history_copy.insert(
                0, AgentChatHistory(message=EasyInputMessageParam(role="system", content=system_prompt))
            )

            # Collect tool definitions from all tools
            request_tools = [defn for tool in turn_config.tools for defn in tool.TOOLS.values()]
            if task_tool is not None:
                request_tools.extend(task_tool.TOOLS.values())

            # Add built-in web search tool
            if turn_config.enable_built_in_web_tool:
                request_tools.append(WebSearchToolParam(type="web_search"))

            # Convert to ChatMessage for router (strips subagents)
            input_messages: list[ChatMessage] = [
                ChatMessage(
                    message=msg.message,
                    id=msg.id,
                    timestamp=msg.timestamp,
                    created_by=msg.created_by,
                    interop=msg.interop,
                    metadata=msg.metadata,
                    provider_kwargs=msg.provider_kwargs,
                    original_response=msg.original_response,
                )
                for msg in history_copy
            ]
            response = await self.router.create(
                input=input_messages,
                model=turn_config.model,
                reasoning=turn_config.reasoning.model_dump(exclude_none=True),
                include=["reasoning.encrypted_content", "web_search_call.results", "web_search_call.action.sources"],
                tools=request_tools,
                max_output_tokens=120_000,
            )

            # Process response messages, add to history, and save
            await self._process_response_messages(response.output, task_tool, tool_by_name)

            # Yield events for the response
            has_tool_calls = False
            for event in self._get_response_events(response.output):
                if isinstance(event, AgentToolCallEvent):
                    has_tool_calls = True
                yield event

            # If no tool calls, we're done
            if not has_tool_calls:
                yield AgentTurnEnd(reason="completed", agent_id=self.agent_id, agent_name=self._agent_name)
                return

            # Execute approved/denied tools
            async for event in self._execute_tools(task_tool, tool_by_name):
                yield event

            # Check if we need to wait for user decisions
            if self._has_pending_tool_calls():
                yield AgentTurnEnd(reason="tools_need_decision", agent_id=self.agent_id, agent_name=self._agent_name)
                return

            # Continue loop to process tool outputs

    def _create_task_tool(self, turn_config: TurnConfig) -> TaskTool | None:
        """Create TaskTool if subagents are configured and depth allows."""
        if not turn_config.subagents:
            return None
        if self.config.max_subagent_depth <= 0:
            return None
        return TaskTool(
            agent_class=Agent,
            agent_config=self.config,
            router=self.router,
            subagents=turn_config.subagents,
            parent_agent=self,
        )

    def _update_tool_permission(self, call_id: str, permission: str) -> None:
        """Update metadata.permission_status for a tool call."""
        for msg in self.history:
            if msg.message.get("type") == "function_call" and msg.message.get("call_id") == call_id:
                msg.metadata["permission_status"] = PERMISSION_TO_STATUS[permission]
                break

    def _has_pending_tool_calls(self) -> bool:
        """Check if any function_call messages have permission_status='pending'."""
        for msg in self.history:
            if msg.message.get("type") == "function_call" and msg.metadata.get("permission_status") == "pending":
                return True
        return False

    def _has_corresponding_output(self, call_id: str) -> bool:
        """Check if a tool output already exists in history for this call_id."""
        for msg in self.history:
            if msg.message.get("type") == "function_call_output" and msg.message.get("call_id") == call_id:
                return True
        return False

    def _check_tool_constraint(
        self, msg: ChatMessage, task_tool: TaskTool | None, tool_by_name: dict[str, Tool]
    ) -> ConstraintPolicy:
        name = msg.message.get("name", "")
        arguments = json.loads(msg.message.get("arguments", "{}"))

        # Check if it's a task tool call
        if task_tool is not None and name in task_tool.TOOLS:
            return ConstraintPolicy.ALLOW

        # Look up the tool by name
        tool = tool_by_name.get(name)
        if tool is None:
            return ConstraintPolicy.ASK

        return tool.check_constraint(**arguments)

    async def _process_response_messages(
        self, messages: list[ChatMessage], task_tool: TaskTool | None, tool_by_name: dict[str, Tool]
    ) -> None:
        """Set permission metadata on tool calls, add to history, and save."""
        for msg in messages:
            if msg.message.get("type") == "function_call":
                policy = self._check_tool_constraint(msg, task_tool, tool_by_name)
                msg.metadata["permission_status"] = POLICY_TO_STATUS[policy]

        self.history.extend(
            AgentChatHistory(
                message=msg.message,
                id=msg.id,
                timestamp=msg.timestamp,
                created_by=msg.created_by,
                interop=msg.interop,
                metadata=msg.metadata,
                provider_kwargs=msg.provider_kwargs,
                original_response=msg.original_response,
            )
            for msg in messages
        )
        await self._save_history()

    def _get_response_events(self, messages: list[ChatMessage]) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        for msg in messages:
            if msg.message.get("type") == "message":
                for content in msg.message.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "output_text":
                        events.append(
                            AgentMessageEvent(
                                message=content.get("text", ""),
                                agent_id=self.agent_id,
                                agent_name=self._agent_name,
                            )
                        )

            elif msg.message.get("type") == "reasoning":
                summary = msg.message.get("summary", [])
                combined_text = "\n".join(s.get("text", "") for s in summary if s.get("text"))
                if combined_text:
                    events.append(
                        AgentReasoningEvent(
                            message=combined_text,
                            agent_id=self.agent_id,
                            agent_name=self._agent_name,
                        )
                    )

            elif msg.message.get("type") == "function_call":
                needs_approval = msg.metadata.get("permission_status") == "pending"
                events.append(
                    AgentToolCallEvent(
                        call_id=msg.message.get("call_id", ""),
                        name=msg.message.get("name", ""),
                        arguments=msg.message.get("arguments", ""),
                        needs_approval=needs_approval,
                        agent_id=self.agent_id,
                        agent_name=self._agent_name,
                    )
                )
        return events

    async def _execute_tools(
        self, task_tool: TaskTool | None, tool_by_name: dict[str, Tool]
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute approved tools and generate denied outputs. Skips pending.
        Has special handling for TaskTool sub-agents.
        """
        for msg in self.history:
            if msg.message.get("type") != "function_call":
                continue

            call_id = msg.message.get("call_id", "")
            if self._has_corresponding_output(call_id):
                continue

            status = msg.metadata.get("permission_status")
            if status not in ("approved", "denied"):
                continue

            name = msg.message.get("name", "")
            if status == "denied":
                output = "The execution of this tool was denied by the user"
            else:
                arguments = json.loads(msg.message.get("arguments", "{}"))
                output = "Default output. This is indicative of an unknown error in executing the tool."
                # The task tool gets special handling since it can yield multiple events
                if task_tool and name == task_tool.TOOL_NAME:
                    events: list[AgentEvent] = []
                    async for event in task_tool.execute(**arguments):
                        yield event
                        events.append(event)
                        if isinstance(event, AgentTurnEnd):
                            # Find the final AgentMessageEvent from the sub-agent. This will be the tool output.
                            final_message_event = next(
                                (e for e in reversed(events) if isinstance(e, AgentMessageEvent)), None
                            )
                            output = (
                                final_message_event.message
                                if final_message_event
                                else "The sub-agent completed without generating a final message. This might indicate an error occurred."
                            )
                            break
                else:
                    output = await self._execute_tool_impl(tool_by_name[name], arguments)

            tool_output_event = AgentToolOutputEvent(
                call_id=call_id,
                name=name,
                output={"result": output},
                agent_id=self.agent_id,
                agent_name=self._agent_name,
            )
            yield tool_output_event

            output_msg = AgentChatHistory(
                message=FunctionCallOutput(call_id=call_id, type="function_call_output", output=output)
            )
            self.history.append(output_msg)
            await self._save_history()

    async def _execute_tool_impl(self, tool: Tool, arguments: dict[str, Any]) -> str:
        """Execute a specific tool with its arguments."""
        result = tool.execute(**arguments)
        if inspect.iscoroutine(result):
            result = await result
        # Handle case where result is a list (e.g., ReadTool returning images)
        if isinstance(result, list):
            return json.dumps(result)
        return result

    def _resolve_chat_file(self) -> Path:
        """Return chat file path from config, or generate one in temp directory."""
        if self.config.chat_file is not None:
            return self.config.chat_file

        temp_dir = Path(tempfile.gettempdir()) / "agent_core_chats"
        temp_dir.mkdir(parents=True, exist_ok=True)

        working_dir_name = self.config.working_dir.name
        sanitized_name = re.sub(r'[<>:"/\\|?*\s]', "_", working_dir_name)

        date_str = datetime.now().strftime("%Y-%m-%d")
        short_uuid = str(uuid.uuid4())[:8]
        filename = f"{sanitized_name}_{date_str}_{short_uuid}.json"

        return temp_dir / filename

    def _load_history(self) -> list[AgentChatHistory]:
        try:
            data = json.loads(self._chat_file.read_text())
        except Exception:
            return []
        return [self._deserialize_history_item(item) for item in data]

    def _deserialize_history_item(self, item: dict[str, Any]) -> AgentChatHistory:
        base = ChatMessage.from_json(json.dumps(item))
        subagents_data = item.get("subagents", [])
        return AgentChatHistory(
            message=base.message,
            id=base.id,
            timestamp=base.timestamp,
            created_by=base.created_by,
            interop=base.interop,
            metadata=base.metadata,
            provider_kwargs=base.provider_kwargs,
            original_response=base.original_response,
            subagents=[self._deserialize_history_item(s) for s in subagents_data],
        )

    async def _save_history(self) -> None:
        def serialize(msg: AgentChatHistory) -> dict[str, Any]:
            data = json.loads(msg.model_dump_json())
            data["subagents"] = [serialize(s) for s in msg.subagents]
            return data

        json_data = json.dumps([serialize(m) for m in self.history], indent=2)
        self._chat_file.write_text(json_data)
