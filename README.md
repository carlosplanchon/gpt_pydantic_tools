# GPT Pydantic Tools

The `gpt_pydantic_tools` repository offers a Python module designed to integrate Pydantic models with GPT-style tool schemas. It facilitates the transformation of Pydantic models into a format that is suitable for use with GPT-4's tools functionality, ensuring structured data and tool interaction consistency.

## Features

- **Model Conversion:** Convert Pydantic models into GPT-4 tool schemas.
- **Flexible Tool Choice Handling:** Provides several strategies for tool invocation such as auto, required, and none, based on the context or specific requirements.

## Installation

Install directly using pip:
```bash
pip install gpt-pydantic-tools
```

## Usage

1. **Define Pydantic Models:** Create your Pydantic models as per your requirements.
2. **Convert to GPT Tool Schema:** Use the `ToolSchemaManager` to convert your Pydantic models into GPT tool schemas.
3. **Tool Choice Management:** Utilize the `get_tool_choice_dict` to manage how tools are chosen for execution based on the defined strategies.

Example usage:
```python
from pydantic import BaseModel
from gpt_pydantic_tools import ToolSchemaManager, get_tool_choice_dict, ToolChoiceEnum

class MyModel(BaseModel):
    name: str
    age: int

# Convert Pydantic model to GPT tool schema
schema_manager = ToolSchemaManager(pydantic_obj=MyModel)
tool_schema = schema_manager.tools_schema

# Determine tool invocation strategy
tool_choice = get_tool_choice_dict(ToolChoiceEnum.AUTO, schema_manager)
```

If you want to do the Pydantic Schema conversion yourself, and directly provide the JSON instead, you can do:
```
schema_manager = ToolSchemaManager(pydantic_obj_json_schema=my_model_json_schema)
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests, create issues, and suggest improvements to the repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
