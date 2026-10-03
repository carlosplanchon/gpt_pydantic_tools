#!/usr/bin/env python3

from pydantic import BaseModel

import jsonschema
from jsonschema.exceptions import ValidationError

from dataclasses import dataclass
from dataclasses import field

import copy

import re

from collections.abc import Iterable

from typing import Any
from typing import Generic
from typing import Optional

from typing_extensions import TypeVar

from .formats import ToolChoiceEnum
from .formats import ToolChoiceT
from .formats import ToolFormat
from .formats import format_tool
from .formats import format_tool_choice
from .formats import group_tools

__all__ = [
    "ToolChoiceEnum",
    "ToolChoiceT",
    "ToolFormat",
    "ToolSchemaManager",
    "ToolsSchemaT",
    "format_tools",
    "get_tool_choice_dict",
    "pydantic_obj_to_tool_schema",
    "remove_key_from_dict",
    "remove_keyword_from_schema",
]


def remove_key_from_dict(
    dict_obj: dict[str, Any],
    key_to_remove: str
        ) -> dict[str, Any]:
    """
    Recursively remove the specified key from a dictionary.

    :param dict_obj: Dictionary from which the key should be removed
    :param key_to_remove: Key that needs to be removed
    :return: Dictionary without the specified key
    """
    if isinstance(dict_obj, dict):
        return {
            k: remove_key_from_dict(
                dict_obj=v, key_to_remove=key_to_remove
            ) for k, v in dict_obj.items() if k != key_to_remove
        }
    elif isinstance(dict_obj, list):
        return [
            remove_key_from_dict(
                dict_obj=item, key_to_remove=key_to_remove
            ) for item in dict_obj
        ]
    else:
        return dict_obj


# JSON Schema keywords whose value is a subschema or a list of subschemas.
SUBSCHEMA_KEYWORDS: frozenset[str] = frozenset({
    "additionalItems", "additionalProperties", "allOf", "anyOf",
    "contains", "else", "if", "items", "not", "oneOf", "prefixItems",
    "propertyNames", "then", "unevaluatedItems", "unevaluatedProperties"
})

# JSON Schema keywords whose value maps names (field names, model
# names, patterns...) to subschemas.
SUBSCHEMA_MAP_KEYWORDS: frozenset[str] = frozenset({
    "$defs", "definitions", "dependentSchemas", "patternProperties",
    "properties"
})


def remove_keyword_from_schema(
    schema: Any,
    keyword: str
        ) -> Any:
    """
    Recursively remove the specified keyword from a JSON schema.

    Unlike remove_key_from_dict, it only walks through subschemas, so
    names (e.g. a field called "title" in "properties") and values
    (e.g. "default", "enum" or "examples") are left untouched.

    :param schema: JSON schema from which the keyword should be removed
    :param keyword: Keyword that needs to be removed
    :return: JSON schema without the specified keyword
    """
    if isinstance(schema, list):
        return [
            remove_keyword_from_schema(schema=item, keyword=keyword)
            for item in schema
        ]
    if not isinstance(schema, dict):
        return schema

    clean_schema: dict[str, Any] = {}
    for key, value in schema.items():
        if key == keyword:
            continue
        if key in SUBSCHEMA_KEYWORDS:
            value = remove_keyword_from_schema(schema=value, keyword=keyword)
        elif key in SUBSCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            value = {
                name: remove_keyword_from_schema(
                    schema=subschema, keyword=keyword
                ) for name, subschema in value.items()
            }
        clean_schema[key] = value
    return clean_schema


###########################################
#                                         #
#   --- PYDANTIC OBJ TO TOOL SCHEMA ---   #
#                                         #
###########################################
ToolsSchemaT = list[dict[str, Any]]

# Function names accepted by the Chat Completions API.
TOOL_NAME_PATTERN = re.compile(r"[a-zA-Z0-9_-]{1,64}")


def pydantic_obj_to_tool_schema(
    pydantic_obj: type[BaseModel] | None = None,
    pydantic_obj_json_schema: dict[Any, Any] | None = None,
    description: str = "",
    tool_name: str | None = None
        ) -> ToolsSchemaT:

    if pydantic_obj is not None and pydantic_obj_json_schema is not None:
        raise ValueError(
            "Pass either pydantic_obj or pydantic_obj_json_schema, not both"
        )

    if pydantic_obj is not None and not (
        isinstance(pydantic_obj, type) and issubclass(pydantic_obj, BaseModel)
    ):
        raise ValueError(
            "pydantic_obj must be a Pydantic model class, "
            f"got {pydantic_obj!r}"
        )

    if pydantic_obj_json_schema is not None and not isinstance(
        pydantic_obj_json_schema, dict
    ):
        raise ValueError(
            "pydantic_obj_json_schema must be a dict, "
            f"got {pydantic_obj_json_schema!r}"
        )

    if not isinstance(description, str):
        raise ValueError(f"description must be a str, got {description!r}")

    if pydantic_obj is not None:
        json_data = pydantic_obj.model_json_schema()
    elif pydantic_obj_json_schema is not None:
        json_data = pydantic_obj_json_schema
    else:
        raise ValueError(
            "You need to provide either pydantic_obj "
            "or pydantic_obj_json_schema"
        )

    if tool_name is None:
        if "title" not in json_data:
            raise ValueError(
                "The JSON schema has no title to name the tool after, "
                "pass tool_name"
            )
        tool_name = json_data["title"]

    if not (
        isinstance(tool_name, str) and TOOL_NAME_PATTERN.fullmatch(tool_name)
    ):
        raise ValueError(
            f"Invalid tool name {tool_name!r}: it must have 1 to 64 ASCII "
            "letters, digits, underscores or dashes (pass tool_name to "
            "choose another one)"
        )

    parameters = remove_keyword_from_schema(
        schema=json_data,
        keyword="title"
    )

    # The schema description (e.g. the model docstring) is only used when
    # no description is given.
    schema_description = parameters.pop("description", "")
    description = description or schema_description

    tool_dict = {
        "name": tool_name,
        "description": description,
        "parameters": parameters
    }

    tools_schema = [
        {
            "type": "function",
            "function": tool_dict
        }
    ]
    return tools_schema


# Model of a manager. Covariant, so that a manager of any model fits where
# a ToolSchemaManager is expected, and BaseModel by default, for managers
# built from a JSON schema.
ModelT = TypeVar("ModelT", bound=BaseModel, covariant=True, default=BaseModel)


class _WeakReferenceable:
    # Gives instances the __weakref__ slot. dataclass(weakref_slot=True)
    # leaves it out on Python 3.12.0 to 3.12.3 when the class also
    # subclasses Generic (CPython gh-118033, fixed in 3.12.4).
    __slots__ = ("__weakref__",)


@dataclass(slots=True)
class ToolSchemaManager(_WeakReferenceable, Generic[ModelT]):
    pydantic_obj: Optional[type[ModelT]] = None
    pydantic_obj_json_schema: Optional[dict[Any, Any]] = None

    tools_schema: ToolsSchemaT = field(init=False)

    description: str = ""

    # When not given, the tool is named after the schema title.
    tool_name: Optional[str] = None

    def __post_init__(self) -> None:
        self.tools_schema = pydantic_obj_to_tool_schema(
            pydantic_obj=self.pydantic_obj,
            pydantic_obj_json_schema=self.pydantic_obj_json_schema,
            description=self.description,
            tool_name=self.tool_name
        )

        self.tool_name = self.tools_schema[0]["function"]["name"]

    def validate(self, arguments: Any) -> None:
        """
        Check the arguments of a tool call against the tool schema.

        :param arguments: Arguments of the tool call, already parsed
        :raises jsonschema.exceptions.ValidationError: If they don't match
        """
        # The tool parameters exist no matter how the manager was built
        # (from pydantic_obj or from pydantic_obj_json_schema).
        jsonschema.validate(
            instance=arguments,
            schema=self.tools_schema[0]["function"]["parameters"]
        )

    def is_valid(self, arguments: Any) -> bool:
        """
        Tell whether the arguments of a tool call match the tool schema.

        :param arguments: Arguments of the tool call, already parsed
        :return: True if they match, False otherwise
        """
        try:
            self.validate(arguments)
        except ValidationError:
            return False
        return True

    def parse(self, arguments: Any) -> ModelT:
        """
        Turn the arguments of a tool call into an instance of the model.

        Unlike validate(), it runs the Pydantic model, with its validators
        and type conversions.

        :param arguments: Arguments of the tool call, as a JSON string
            (OpenAI) or already parsed (Anthropic, Gemini)
        :return: Instance of the model
        :raises pydantic.ValidationError: If they don't fit the model
        """
        if self.pydantic_obj is None:
            raise ValueError(
                "parse() needs a Pydantic model, "
                "build the manager with pydantic_obj"
            )
        if isinstance(arguments, (str, bytes, bytearray)):
            return self.pydantic_obj.model_validate_json(arguments)
        return self.pydantic_obj.model_validate(arguments)

    def validate_tool_answer(
        self,
        schema_to_validate: Any,
            ) -> bool | ValidationError:

        try:
            self.validate(schema_to_validate)
            # print("JSON data is valid.")
            return True
        except ValidationError as err:
            # print("JSON data is invalid.")
            return err

    def tool(
        self,
        tool_format: ToolFormat = ToolFormat.CHAT_COMPLETIONS
            ) -> dict[str, Any]:
        """
        Return the tool definition in the format of the given API.

        :param tool_format: Format of the API that receives the tool
        :return: Tool definition, a copy that can be changed freely
        """
        function = self.tools_schema[0]["function"]
        return format_tool(
            name=function["name"],
            description=function["description"],
            parameters=copy.deepcopy(function["parameters"]),
            tool_format=tool_format
        )


def format_tools(
    schema_managers: Iterable[ToolSchemaManager],
    tool_format: ToolFormat = ToolFormat.CHAT_COMPLETIONS
        ) -> list[dict[str, Any]]:
    """
    Build the tools list of a request from several managers.

    :param schema_managers: Managers of the tools to send
    :param tool_format: Format of the API that receives the tools
    :return: Value for the tools parameter of the API
    """
    managers = list(schema_managers)

    # The model calls tools by name, so two tools can't share one.
    names: set[str] = set()
    for manager in managers:
        name = manager.tools_schema[0]["function"]["name"]
        if name in names:
            raise ValueError(f"Duplicate tool name: {name!r}")
        names.add(name)

    return group_tools(
        tools=[manager.tool(tool_format) for manager in managers],
        tool_format=tool_format
    )


#####################################
#                                   #
#   --- TOOL CHOICE PARAMETER ---   #
#                                   #
#####################################
def get_tool_choice_dict(
    tool_choice: ToolChoiceEnum,
    schema_manager: ToolSchemaManager,
    tool_format: ToolFormat = ToolFormat.CHAT_COMPLETIONS
        ) -> ToolChoiceT:
    return format_tool_choice(
        tool_choice=tool_choice,
        tool_name=schema_manager.tools_schema[0]["function"]["name"],
        tool_format=tool_format
    )
