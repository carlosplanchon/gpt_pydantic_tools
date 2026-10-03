# GPT Pydantic Tools

Turn Pydantic models into function-calling tools. `gpt-pydantic-tools` converts a model, or a JSON schema, into the tool format of the main LLM APIs (OpenAI Chat Completions and Responses, Anthropic and Gemini), builds the matching `tool_choice` value, and validates the arguments the model sends back.

[![CI](https://github.com/carlosplanchon/gpt-pydantic-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/carlosplanchon/gpt-pydantic-tools/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/gpt-pydantic-tools.svg)](https://pypi.org/project/gpt-pydantic-tools/)
[![Python versions](https://img.shields.io/pypi/pyversions/gpt-pydantic-tools.svg)](https://pypi.org/project/gpt-pydantic-tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/carlosplanchon/gpt-pydantic-tools)

## Features

- **Model to tool:** converts a Pydantic model, or a JSON schema, into a function tool for OpenAI (Chat Completions or Responses), Anthropic or Gemini. The model's docstring becomes the tool description, unless you pass `description`.
- **Lean schemas:** strips the `title` that Pydantic adds to every schema, without touching fields or values that are also called `title`.
- **Valid tool names:** names the tool after the model, or after `tool_name`, and checks the name against the API rules: 1 to 64 ASCII letters, digits, underscores or dashes.
- **`tool_choice` values:** `auto`, `required`, `none`, or forcing this tool, in the format of each API.
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

### Other formats

`tools_schema` holds the tool in the Chat Completions format. `tool()` returns it in the format of another API, and `get_tool_choice_dict()` takes the same format as its third argument:

| `ToolFormat` | API | Pass `tool()` in | Pass `get_tool_choice_dict()` as |
|---|---|---|---|
| `CHAT_COMPLETIONS` (default) | OpenAI Chat Completions, and compatible APIs such as Mistral's and Ollama's | `tools` | `tool_choice` |
| `RESPONSES` | OpenAI Responses API | `tools` | `tool_choice` |
| `ANTHROPIC` | Anthropic's Messages API (Claude) | `tools` | `tool_choice` |
| `GEMINI` | Gemini's `generate_content` | `tools=[{"function_declarations": [...]}]` | `tool_config` |
| `GEMINI_INTERACTIONS` | Gemini's Interactions API | `tools` | `generation_config["tool_choice"]` |

With Anthropic's SDK:

```python
import anthropic

from gpt_pydantic_tools import ToolFormat

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    messages=[{"role": "user", "content": "What's the weather in Montevideo?"}],
    tools=[weather_tool.tool(ToolFormat.ANTHROPIC)],
    tool_choice=get_tool_choice_dict(ToolChoiceEnum.AUTO, weather_tool, ToolFormat.ANTHROPIC),
)
for block in message.content:
    if block.type == "tool_use":
        weather_tool.validate(block.input)
```

where `weather_tool.tool(ToolFormat.ANTHROPIC)` is:

```json
{
  "name": "get_weather",
  "description": "Get the current weather in a city.",
  "input_schema": {
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
```

With Gemini's `generate_content`, group the tools of every manager in a single `function_declarations` list:

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents="What's the weather in Montevideo?",
    config=types.GenerateContentConfig(
        tools=[{"function_declarations": [weather_tool.tool(ToolFormat.GEMINI)]}],
        tool_config=get_tool_choice_dict(ToolChoiceEnum.TOOL_NAME, weather_tool, ToolFormat.GEMINI),
    ),
)
weather_tool.validate(response.function_calls[0].args)
```

## Compatibility

The tools are plain dicts, so they work with each provider's SDK or with raw HTTP requests. Keep in mind:

- The schemas are not prepared for strict mode (`additionalProperties: false` on every object and every field in `required`). If you use the OpenAI Python SDK and need strict mode, `openai.pydantic_function_tool()` covers that case.
- The newest Claude models (Opus 5.5, Sonnet 5.5 and Fable 5.1) reject forcing a tool, so `REQUIRED` and `TOOL_NAME` fail there. Use `AUTO` and ask for the tool in the prompt.
- Gemini asks for a description of every function, and supports a subset of JSON Schema.
- Tool names follow the strictest rule of these APIs (Chat Completions'), so a valid name works with all of them.

`gpt-pydantic-tools` only depends on Pydantic and jsonschema, which makes it handy when you build the requests yourself or switch between providers.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, create issues, and suggest improvements to the repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
