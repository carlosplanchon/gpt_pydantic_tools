from enum import StrEnum

from typing import Any


ToolChoiceT = dict[str, Any] | str


class ToolChoiceEnum(StrEnum):
    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"
    TOOL_NAME = "tool_name"


class ToolFormat(StrEnum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GEMINI_INTERACTIONS = "gemini_interactions"


def format_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    tool_format: ToolFormat
        ) -> dict[str, Any]:
    """
    Build a tool definition in the format of the given API.

    :param name: Name of the tool
    :param description: Description of the tool
    :param parameters: JSON schema of the tool arguments
    :param tool_format: Format of the API that receives the tool
    :return: Tool definition
    """
    match tool_format:
        case ToolFormat.CHAT_COMPLETIONS:
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            }
        case ToolFormat.RESPONSES:
            # The OpenAI SDK types require the strict key; None leaves it
            # unset.
            return {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters,
                "strict": None
            }
        case ToolFormat.GEMINI_INTERACTIONS:
            return {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters
            }
        case ToolFormat.ANTHROPIC:
            return {
                "name": name,
                "description": description,
                "input_schema": parameters
            }
        case ToolFormat.GEMINI:
            # A function declaration: Gemini tools group them in
            # {"function_declarations": [...]}.
            return {
                "name": name,
                "description": description,
                "parameters_json_schema": parameters
            }
        case _:
            raise ValueError(f"Invalid tool format: {tool_format}")


def group_tools(
    tools: list[dict[str, Any]],
    tool_format: ToolFormat
        ) -> list[dict[str, Any]]:
    """
    Build the tools list of a request from several tool definitions.

    Gemini's generate_content groups the function declarations in a
    single tool; the other APIs take the tool definitions as they are.

    :param tools: Tool definitions in the given format
    :param tool_format: Format of the API that receives the tools
    :return: Value for the tools parameter of the API
    """
    match tool_format:
        case ToolFormat.GEMINI:
            return [{"function_declarations": tools}] if tools else []
        case (
            ToolFormat.CHAT_COMPLETIONS
            | ToolFormat.RESPONSES
            | ToolFormat.ANTHROPIC
            | ToolFormat.GEMINI_INTERACTIONS
        ):
            return tools
        case _:
            raise ValueError(f"Invalid tool format: {tool_format}")


def format_tool_choice(
    tool_choice: ToolChoiceEnum,
    tool_name: str,
    tool_format: ToolFormat
        ) -> ToolChoiceT:
    """
    Build the value that tells the API how to choose tools.

    For Gemini's generate_content it is the tool_config, and for Gemini's
    Interactions API, the tool_choice of the generation_config.

    :param tool_choice: How the model should choose tools
    :param tool_name: Name of the tool, used by TOOL_NAME
    :param tool_format: Format of the API that receives the value
    :return: Value in the format of the API
    """
    choices: dict[ToolChoiceEnum, ToolChoiceT]
    match tool_format:
        case ToolFormat.CHAT_COMPLETIONS:
            choices = {
                ToolChoiceEnum.AUTO: "auto",
                ToolChoiceEnum.REQUIRED: "required",
                ToolChoiceEnum.NONE: "none",
                ToolChoiceEnum.TOOL_NAME: {
                    "type": "function",
                    "function": {"name": tool_name}
                }
            }
        case ToolFormat.RESPONSES:
            choices = {
                ToolChoiceEnum.AUTO: "auto",
                ToolChoiceEnum.REQUIRED: "required",
                ToolChoiceEnum.NONE: "none",
                ToolChoiceEnum.TOOL_NAME: {
                    "type": "function",
                    "name": tool_name
                }
            }
        case ToolFormat.ANTHROPIC:
            choices = {
                ToolChoiceEnum.AUTO: {"type": "auto"},
                ToolChoiceEnum.REQUIRED: {"type": "any"},
                ToolChoiceEnum.NONE: {"type": "none"},
                ToolChoiceEnum.TOOL_NAME: {"type": "tool", "name": tool_name}
            }
        case ToolFormat.GEMINI:
            choices = {
                ToolChoiceEnum.AUTO: {
                    "function_calling_config": {"mode": "AUTO"}
                },
                ToolChoiceEnum.REQUIRED: {
                    "function_calling_config": {"mode": "ANY"}
                },
                ToolChoiceEnum.NONE: {
                    "function_calling_config": {"mode": "NONE"}
                },
                ToolChoiceEnum.TOOL_NAME: {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [tool_name]
                    }
                }
            }
        case ToolFormat.GEMINI_INTERACTIONS:
            choices = {
                ToolChoiceEnum.AUTO: {"allowed_tools": {"mode": "auto"}},
                ToolChoiceEnum.REQUIRED: {"allowed_tools": {"mode": "any"}},
                ToolChoiceEnum.NONE: {"allowed_tools": {"mode": "none"}},
                ToolChoiceEnum.TOOL_NAME: {
                    "allowed_tools": {"mode": "any", "tools": [tool_name]}
                }
            }
        case _:
            raise ValueError(f"Invalid tool format: {tool_format}")

    if tool_choice not in choices:
        raise ValueError(f"Invalid tool choice: {tool_choice}")
    return choices[tool_choice]
