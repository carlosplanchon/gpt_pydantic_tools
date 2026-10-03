import copy
import pickle
import weakref
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

import pytest
from jsonschema import ValidationError
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from gpt_pydantic_tools import (
    ToolChoiceEnum,
    ToolSchemaManager,
    get_tool_choice_dict,
    pydantic_obj_to_tool_schema,
    remove_key_from_dict,
    remove_keyword_from_schema,
)


class MyModel(BaseModel):
    name: str
    age: int


class Book(BaseModel):
    """A book."""
    title: str
    pages: int


class Author(BaseModel):
    title: str | None = None
    name: str


class Library(BaseModel):
    books: list[Book]
    author: Author | None = None
    meta: dict = Field(default={"title": "x"})
    extra: dict = Field(default_factory=dict, examples=[{"title": "Dune"}])


def parameters(manager: ToolSchemaManager) -> dict:
    return manager.tools_schema[0]["function"]["parameters"]


@pytest.fixture(params=["pydantic_obj", "pydantic_obj_json_schema"])
def make_manager(request):
    """Build a ToolSchemaManager in each of the two supported ways."""
    def make(model: type[BaseModel], **kwargs) -> ToolSchemaManager:
        if request.param == "pydantic_obj":
            return ToolSchemaManager(pydantic_obj=model, **kwargs)
        return ToolSchemaManager(
            pydantic_obj_json_schema=model.model_json_schema(), **kwargs
        )
    return make


# --- Tool schema ---

def test_tool_schema_from_readme_example():
    schema_manager = ToolSchemaManager(pydantic_obj=MyModel)

    assert schema_manager.tool_name == "MyModel"
    assert schema_manager.tools_schema == [
        {
            "type": "function",
            "function": {
                "name": "MyModel",
                "description": "",
                "parameters": {
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                    "type": "object",
                },
            },
        }
    ]


def test_description_comes_from_the_docstring():
    function = ToolSchemaManager(pydantic_obj=Book).tools_schema[0]["function"]

    assert function["description"] == "A book."
    assert "description" not in function["parameters"]


def test_function_and_manager_give_the_same_tool():
    from_function = pydantic_obj_to_tool_schema(pydantic_obj=MyModel)
    from_manager = ToolSchemaManager(pydantic_obj=MyModel).tools_schema

    assert from_function == from_manager
    assert from_function[0]["function"]["description"] == ""


def test_function_rejects_a_description_that_is_not_a_str():
    with pytest.raises(ValueError, match="description"):
        pydantic_obj_to_tool_schema(pydantic_obj=MyModel, description=None)


def test_requires_a_model_or_a_json_schema():
    with pytest.raises(ValueError, match="pydantic_obj"):
        ToolSchemaManager()


def test_rejects_both_a_model_and_a_json_schema():
    with pytest.raises(ValueError, match="not both"):
        ToolSchemaManager(
            pydantic_obj=MyModel,
            pydantic_obj_json_schema=MyModel.model_json_schema(),
        )


@pytest.mark.parametrize(
    "value",
    [Book(title="Dune", pages=412), dict],
    ids=["model-instance", "class-that-is-not-a-model"],
)
def test_pydantic_obj_must_be_a_model_class(value):
    with pytest.raises(ValueError):
        ToolSchemaManager(pydantic_obj=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pydantic_obj_json_schema": "not a dict"},
        {"pydantic_obj": Book, "description": None},
        {"pydantic_obj": Book, "description": 1},
    ],
    ids=["json-schema-not-a-dict", "description-none", "description-not-a-str"],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        ToolSchemaManager(**kwargs)


def test_manager_keeps_the_behavior_attrs_gave_it():
    manager = ToolSchemaManager(MyModel)  # positional, as with attrs

    assert manager == ToolSchemaManager(pydantic_obj=MyModel)
    assert copy.copy(manager) == manager
    assert copy.deepcopy(manager) == manager
    assert pickle.loads(pickle.dumps(manager)) == manager
    assert weakref.ref(manager)() is manager
    assert not hasattr(manager, "__dict__")
    with pytest.raises(TypeError):
        hash(manager)


def test_field_named_title_is_kept():
    assert parameters(ToolSchemaManager(pydantic_obj=Book)) == {
        "properties": {
            "title": {"type": "string"},
            "pages": {"type": "integer"},
        },
        "required": ["title", "pages"],
        "type": "object",
    }


def test_nested_models_keep_their_title_fields():
    defs = parameters(ToolSchemaManager(pydantic_obj=Library))["$defs"]

    assert "title" in defs["Book"]["properties"]
    assert "title" in defs["Author"]["properties"]
    assert all("title" not in model for model in defs.values())


def test_literal_values_are_left_untouched():
    properties = parameters(ToolSchemaManager(pydantic_obj=Library))["properties"]

    assert properties["meta"]["default"] == {"title": "x"}
    assert properties["extra"]["examples"] == [{"title": "Dune"}]


def test_only_title_annotations_are_removed():
    # A "title" annotation in every subschema position, next to names and
    # literal values that are also called "title".
    schema = {
        "title": "Root",
        "type": "object",
        "properties": {
            "title": {
                "title": "Title",
                "type": "string",
                "default": "title",
                "examples": [{"title": "t"}],
            },
            "items": {
                "title": "Items",
                "type": "array",
                "items": {"title": "Item", "type": "integer"},
            },
            "any": {
                "title": "Any",
                "anyOf": [
                    {"title": "A", "type": "string"},
                    {"title": "B", "type": "null"},
                ],
            },
            "map": {
                "title": "Map",
                "type": "object",
                "additionalProperties": {"title": "V", "type": "integer"},
            },
            "pattern": {
                "title": "Pattern",
                "type": "object",
                "patternProperties": {
                    "^title$": {"title": "P", "type": "string"},
                },
            },
            "tuple": {
                "title": "Tuple",
                "prefixItems": [{"title": "X", "type": "integer"}],
                "items": False,
            },
            "conditional": {
                "title": "Conditional",
                "if": {"title": "If"},
                "then": {"title": "Then"},
                "else": {"title": "Else"},
                "not": {"title": "Not"},
            },
            "const": {"title": "Const", "const": {"title": "value"}},
            "enum": {"title": "Enum", "enum": [{"title": "a"}, "title"]},
            "properties": {
                "title": "Properties",
                "type": "object",
                "properties": {"title": {"title": "Inner"}},
            },
        },
        "required": ["title"],
        "$defs": {
            "title": {
                "title": "TitleModel",
                "type": "object",
                "properties": {"title": {"title": "Title", "type": "string"}},
            },
        },
        "dependentRequired": {"title": ["items"]},
    }

    assert remove_keyword_from_schema(schema=schema, keyword="title") == {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "default": "title",
                "examples": [{"title": "t"}],
            },
            "items": {"type": "array", "items": {"type": "integer"}},
            "any": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "map": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
            },
            "pattern": {
                "type": "object",
                "patternProperties": {"^title$": {"type": "string"}},
            },
            "tuple": {"prefixItems": [{"type": "integer"}], "items": False},
            "conditional": {"if": {}, "then": {}, "else": {}, "not": {}},
            "const": {"const": {"title": "value"}},
            "enum": {"enum": [{"title": "a"}, "title"]},
            "properties": {"type": "object", "properties": {"title": {}}},
        },
        "required": ["title"],
        "$defs": {
            "title": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
            },
        },
        "dependentRequired": {"title": ["items"]},
    }


def test_input_schema_is_not_mutated():
    schema = Library.model_json_schema()
    original = copy.deepcopy(schema)

    remove_keyword_from_schema(schema=schema, keyword="title")

    assert schema == original


class Color(Enum):
    RED = "red"
    GREEN = "green"


class Cat(BaseModel):
    pet_type: Literal["cat"]
    meows: int


class Dog(BaseModel):
    pet_type: Literal["dog"]
    barks: float


class Node(BaseModel):
    value: int
    children: list["Node"] = []


class Complex(BaseModel):
    """Model without fields, names or values called title."""
    when: datetime
    uid: UUID
    color: Color = Color.RED
    pet: Annotated[Cat | Dog, Field(discriminator="pet_type")]
    maybe_pet: Cat | None = None
    pair: tuple[int, str]
    many: set[int]
    mapping: dict[str, Dog]
    keyed: dict[Annotated[str, StringConstraints(pattern=r"^k")], int]
    tree: Node
    code: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
    snap: int = Field(default=42, title="The Snap", gt=30, lt=50)
    either: int | str
    nums: list[Annotated[int, Field(title="Num", ge=0)]] = []
    free: dict = Field(default_factory=dict, examples=[{"a": 1}])


def test_removes_every_title_pydantic_generates():
    # With nothing else called "title", the result must match a blind
    # recursive removal: no title annotation may be left behind.
    schema = Complex.model_json_schema()

    assert remove_keyword_from_schema(
        schema=schema, keyword="title"
    ) == remove_key_from_dict(dict_obj=schema, key_to_remove="title")


# --- Tool name ---

class WeirdTitle(BaseModel):
    model_config = ConfigDict(title="Get weather (v2)")
    city: str


def test_tool_name_can_be_given(make_manager):
    manager = make_manager(Book, tool_name="get_book")

    assert manager.tool_name == "get_book"
    assert manager.tools_schema[0]["function"]["name"] == "get_book"
    assert get_tool_choice_dict(ToolChoiceEnum.TOOL_NAME, manager) == {
        "type": "function",
        "function": {"name": "get_book"},
    }


def test_title_that_is_not_a_valid_tool_name(make_manager):
    with pytest.raises(ValueError, match="tool_name"):
        make_manager(WeirdTitle)

    manager = make_manager(WeirdTitle, tool_name="get_weather")

    assert manager.tool_name == "get_weather"


def test_json_schema_without_title_needs_a_tool_name():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}

    with pytest.raises(ValueError, match="no title"):
        ToolSchemaManager(pydantic_obj_json_schema=schema)

    manager = ToolSchemaManager(
        pydantic_obj_json_schema=schema, tool_name="get_weather"
    )

    assert manager.tool_name == "get_weather"


@pytest.mark.parametrize("tool_name", ["a", "get-book_2", "a" * 64])
def test_valid_tool_names(tool_name):
    manager = ToolSchemaManager(pydantic_obj=Book, tool_name=tool_name)

    assert manager.tool_name == tool_name


@pytest.mark.parametrize(
    "tool_name",
    ["", "a" * 65, "get book", "Canción", "get.book", 123],
    ids=["empty", "too-long", "space", "non-ascii", "dot", "not-a-str"],
)
def test_invalid_tool_names(tool_name):
    with pytest.raises(ValueError, match="Invalid tool name"):
        ToolSchemaManager(pydantic_obj=Book, tool_name=tool_name)


# --- Tool answer validation ---

def test_valid_answer(make_manager):
    manager = make_manager(Book)
    answer = {"title": "Dune", "pages": 412}

    manager.validate(answer)  # does not raise
    assert manager.is_valid(answer) is True
    assert manager.validate_tool_answer(answer) is True


@pytest.mark.parametrize(
    "answer",
    [{"title": "Dune"}, {"title": 1, "pages": 412}],
    ids=["missing-field", "wrong-type"],
)
def test_invalid_answer(make_manager, answer):
    manager = make_manager(Book)

    with pytest.raises(ValidationError):
        manager.validate(answer)
    assert manager.is_valid(answer) is False
    assert isinstance(manager.validate_tool_answer(answer), ValidationError)


def test_nested_answer_is_validated_through_refs(make_manager):
    manager = make_manager(Library)

    assert manager.validate_tool_answer(
        {"books": [{"title": "Dune", "pages": 412}]}
    ) is True
    assert isinstance(
        manager.validate_tool_answer({"books": [{"pages": 412}]}),
        ValidationError,
    )


# --- Tool choice ---

@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        (ToolChoiceEnum.AUTO, "auto"),
        (ToolChoiceEnum.REQUIRED, "required"),
        (ToolChoiceEnum.NONE, "none"),
        (
            ToolChoiceEnum.TOOL_NAME,
            {"type": "function", "function": {"name": "MyModel"}},
        ),
    ],
)
def test_get_tool_choice_dict(tool_choice, expected):
    schema_manager = ToolSchemaManager(pydantic_obj=MyModel)

    assert get_tool_choice_dict(tool_choice, schema_manager) == expected
