# GPT Pydantic Tools

Turn Pydantic models into function-calling tools. `gpt-pydantic-tools` converts a model, or a JSON schema, into the tool format of the Chat Completions API, builds the matching `tool_choice` value, and validates the arguments the model sends back.

[![CI](https://github.com/carlosplanchon/gpt-pydantic-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/carlosplanchon/gpt-pydantic-tools/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/gpt-pydantic-tools.svg)](https://pypi.org/project/gpt-pydantic-tools/)
[![Python versions](https://img.shields.io/pypi/pyversions/gpt-pydantic-tools.svg)](https://pypi.org/project/gpt-pydantic-tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/carlosplanchon/gpt-pydantic-tools)

## Features

- **Model to tool:** converts a Pydantic model, or a JSON schema, into a Chat Completions function tool. The model's docstring becomes the tool description.
- **Lean schemas:** strips the `title` that Pydantic adds to every schema, without touching fields or values that are also called `title`.
- **Valid tool names:** names the tool after the model, or after `tool_name`, and checks the name against the API rules: 1 to 64 ASCII letters, digits, underscores or dashes.
- **`tool_choice` values:** `auto`, `required`, `none`, or forcing this tool.
- **Answer validation:** checks the arguments the model returns against the tool's schema.

## Installation

```bash
uv add gpt-pydantic-tools
```

Or with pip:

```bash
pip install gpt-pydantic-tools
```

## Usage

### Define a tool

```python
from typing import Literal

from pydantic import BaseModel, Field

from gpt_pydantic_tools import ToolChoiceEnum, ToolSchemaManager, get_tool_choice_dict


class GetWeather(BaseModel):
    """Get the current weather in a city."""

    city: str = Field(description="City name, e.g. Montevideo")
    unit: Literal["celsius", "fahrenheit"] = "celsius"


weather_tool = ToolSchemaManager(pydantic_obj=GetWeather, tool_name="get_weather")
```

Without `tool_name`, the tool is named after the model (`GetWeather`).

`weather_tool.tools_schema` holds the tool, ready to send:

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather in a city.",
      "parameters": {
        "properties": {
          "city": {
            "description": "City name, e.g. Montevideo",
            "type": "string"
          },
          "unit": {
            "default": "celsius",
            "enum": [
              "celsius",
              "fahrenheit"
            ],
            "type": "string"
          }
        },
        "required": [
          "city"
        ],
        "type": "object"
      }
    }
  }
]
```

### Call the API

Pass `tools_schema` as `tools`, and the result of `get_tool_choice_dict()` as `tool_choice`. With the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-6-astra",
    messages=[{"role": "user", "content": "What's the weather in Montevideo?"}],
    tools=weather_tool.tools_schema,
    tool_choice=get_tool_choice_dict(ToolChoiceEnum.TOOL_NAME, weather_tool),
)
tool_call = completion.choices[0].message.tool_calls[0]
```

`ToolChoiceEnum.TOOL_NAME` forces the model to call this tool. `AUTO`, `REQUIRED` and `NONE` map to `"auto"`, `"required"` and `"none"`.

### Validate the answer

The model does not always follow the schema, so check the arguments before using them. `validate()` raises a `jsonschema.ValidationError` that describes the problem:

```python
import json

from jsonschema import ValidationError

arguments = json.loads(tool_call.function.arguments)

try:
    weather_tool.validate(arguments)
except ValidationError as error:
    print(error.message)  # e.g. 'kelvin' is not one of ['celsius', 'fahrenheit']
```

If you only need a yes or no, `weather_tool.is_valid(arguments)` returns `True` or `False`.

`validate_tool_answer()`, from earlier versions, still returns `True` or the error instead of raising it. Compare its result with `is True`: the error counts as true in an `if`.

### Start from a JSON schema

If you already have the JSON schema, pass it instead of the model. The tool is named after the schema's `title`; if it has none, pass `tool_name`:

```python
schema_manager = ToolSchemaManager(
    pydantic_obj_json_schema=my_json_schema,
    tool_name="my_tool",
)
```

## Compatibility

The tools follow the Chat Completions format, which other APIs also accept, such as Mistral's and Ollama's. They are not in the format of:

- OpenAI's Responses API, which puts `name` and `parameters` at the top level of the tool.
- Anthropic's API for Claude, which uses `input_schema`.

The schemas are not prepared for strict mode either (`strict: true`, which requires `additionalProperties: false` and every field in `required`). If you use the OpenAI Python SDK and need strict mode, `openai.pydantic_function_tool()` covers that case. `gpt-pydantic-tools` only depends on Pydantic and jsonschema, which makes it handy when you build the requests yourself or use another provider's client.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, create issues, and suggest improvements to the repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
