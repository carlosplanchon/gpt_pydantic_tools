#!/usr/bin/env python3

import attrs
from attrs_strict import type_validator

from pydantic import BaseModel

import jsonschema

# https://github.com/pydantic/pydantic/issues/6381
# ModelMetaclass is not a public class.
# They want to be able to refactor the ModelMetaclass
# without it being considered a breaking change.
from pydantic._internal._model_construction import ModelMetaclass

from enum import StrEnum

from typing import Any
from typing import Optional


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


def pydantic_obj_to_tool_schema(
    pydantic_obj: BaseModel | None = None,
    pydantic_obj_json_schema: dict[Any, Any] | None = None,
    description: str = None
        ) -> ToolsSchemaT:

    if pydantic_obj is None and pydantic_obj_json_schema is None:
        raise ValueError(
            "You need to provide either pydantic_obj "
            "or pydantic_obj_json_schema"
        )

    if pydantic_obj is not None:
        json_data = pydantic_obj.schema()
    else:
        json_data = pydantic_obj_json_schema

    func_name: str = json_data["title"]

    gpt_function_dict = remove_keyword_from_schema(
        schema=json_data,
        keyword="title"
    )

    if "description" in gpt_function_dict.keys():
        description: str = gpt_function_dict.pop("description")
    else:
        description = description

    tool_dict = {
        "name": func_name,
        "description": description,
        "parameters": gpt_function_dict
    }

    tools_schema = [
        {
            "type": "function",
            "function": tool_dict
        }
    ]
    return tools_schema


@attrs.define()
class ToolSchemaManager:
    pydantic_obj: Optional[ModelMetaclass] = attrs.field(
        validator=type_validator(),
        default=None
    )
    pydantic_obj_json_schema: Optional[dict[Any, Any]] = attrs.field(
        validator=type_validator(),
        default=None
    )

    tools_schema: ToolsSchemaT = attrs.field(
        validator=type_validator(),
        init=False
    )

    tool_name: str = attrs.field(
        validator=type_validator(),
        init=False
    )

    description: str = attrs.field(
        validator=type_validator(),
        default=""
    )

    def __attrs_post_init__(self):
        self.tools_schema = pydantic_obj_to_tool_schema(
            pydantic_obj=self.pydantic_obj,
            pydantic_obj_json_schema=self.pydantic_obj_json_schema,
            description=self.description
        )

        self.tool_name = self.tools_schema[0]["function"]["name"]

    def validate_tool_answer(
        self,
        schema_to_validate,
            ) -> bool | jsonschema.exceptions.ValidationError:

        # The tool parameters exist no matter how the manager was built
        # (from pydantic_obj or from pydantic_obj_json_schema).
        parameters_schema = self.tools_schema[0]["function"]["parameters"]
        try:
            jsonschema.validate(
                instance=schema_to_validate,
                schema=parameters_schema
            )
            # print("JSON data is valid.")
            return True
        except jsonschema.exceptions.ValidationError as err:
            # print("JSON data is invalid.")
            return err


#####################################
#                                   #
#   --- TOOL CHOICE PARAMETER ---   #
#                                   #
#####################################
ToolChoiceT = dict[str, Any] | str


class ToolChoiceEnum(StrEnum):
    AUTO: str = "auto"
    REQUIRED: str = "required"
    NONE: str = "none"
    TOOL_NAME: str = "tool_name"


def get_tool_choice_dict(
    tool_choice: ToolChoiceEnum,
    schema_manager: ToolSchemaManager
        ) -> ToolChoiceT:
    match tool_choice:
        case ToolChoiceEnum.AUTO:
            return "auto"
        case ToolChoiceEnum.REQUIRED:
            return "required"
        case ToolChoiceEnum.NONE:
            return "none"
        case ToolChoiceEnum.TOOL_NAME:
            tool_choice_value = {
                "type": "function",
                "function": {
                    "name": schema_manager.tool_name
                }
            }
        case _:
            raise ValueError(f"Invalid tool choice: {tool_choice}")

    return tool_choice_value
