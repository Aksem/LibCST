# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import dataclass, fields
from typing import Dict, Generator, List, Optional, Sequence, Set, Tuple, Type, Union

import libcst as cst
from libcst import ensure_type, parse_expression
from libcst.codegen.gather import all_libcst_nodes, typeclasses

CST_DIR: Set[str] = set(dir(cst))
CLASS_RE = r"<class \'(.*?)\'>"
OPTIONAL_RE = r"typing\.Union\[([^,]*?), NoneType]"
PASCAL_TO_KEBAB_RE = re.compile(r"(?<!^)(?=[A-Z])")
SIMPLE_STRING_RE = re.compile(r'("\w+"|\'\w+\')')


class CleanseFullTypeNames(cst.CSTTransformer):
    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        # Convert forward ref repr back to a SimpleString.
        if isinstance(updated_node.func, cst.Name) and (
            updated_node.func.deep_equals(cst.Name("_ForwardRef"))
            or updated_node.func.deep_equals(cst.Name("ForwardRef"))
        ):
            return updated_node.args[0].value
        return updated_node

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> Union[cst.Attribute, cst.Name]:
        # Unwrap all attributes, so things like libcst.x.y.Name becomes Name
        return updated_node.attr

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> Union[cst.Name, cst.SimpleString]:
        value = updated_node.value
        if value == "NoneType":
            # This is special-cased in typing, un-special case it.
            return updated_node.with_changes(value="None")
        if value in CST_DIR and not value.endswith("Sentinel"):
            # If this isn't a typing define and it isn't a builtin, convert it to
            # a forward ref string.
            return cst.SimpleString(repr(value))
        return updated_node

    def leave_SubscriptElement(
        self, original_node: cst.SubscriptElement, updated_node: cst.SubscriptElement
    ) -> Union[cst.SubscriptElement, cst.RemovalSentinel]:
        slc = updated_node.slice
        if isinstance(slc, cst.Index):
            val = slc.value
            if isinstance(val, cst.Name):
                if "Sentinel" in val.value:
                    # We don't support maybes in matchers.
                    return cst.RemoveFromParent()
        # Simple trick to kill trailing commas
        return updated_node.with_changes(comma=cst.MaybeSentinel.DEFAULT)


class RemoveTypesFromGeneric(cst.CSTTransformer):
    def __init__(self, values: Sequence[str]) -> None:
        self.values: Set[str] = set(values)

    def leave_SubscriptElement(
        self, original_node: cst.SubscriptElement, updated_node: cst.SubscriptElement
    ) -> Union[cst.SubscriptElement, cst.RemovalSentinel]:
        slc = updated_node.slice
        if isinstance(slc, cst.Index):
            val = slc.value
            if isinstance(val, cst.Name):
                if val.value in self.values:
                    # This type matches, so out it goes
                    return cst.RemoveFromParent()
        return updated_node


def _remove_types(
    oldtype: cst.BaseExpression, values: Sequence[str]
) -> cst.BaseExpression:
    """
    Given a BaseExpression from a type, return a new BaseExpression that does not
    refer to any types listed in values.
    """
    return ensure_type(
        oldtype.visit(RemoveTypesFromGeneric(values)), cst.BaseExpression
    )


class MatcherClassToLibCSTClass(cst.CSTTransformer):
    def leave_SimpleString(
        self, original_node: cst.SimpleString, updated_node: cst.SimpleString
    ) -> Union[cst.SimpleString, cst.Attribute]:
        value = updated_node.evaluated_value
        if value in CST_DIR:
            return cst.Attribute(cst.Name("cst"), cst.Name(value))
        return updated_node


def _convert_match_nodes_to_cst_nodes(
    matchtype: cst.BaseExpression,
) -> cst.BaseExpression:
    """
    Given a BaseExpression in a type, convert this to a new BaseExpression that refers
    to LibCST nodes instead of forward references to matcher nodes.
    """
    return ensure_type(matchtype.visit(MatcherClassToLibCSTClass()), cst.BaseExpression)


def _get_match_if_true(oldtype: cst.BaseExpression) -> cst.SubscriptElement:
    """
    Construct a MatchIfTrue type node appropriate for going into a Union.
    """
    return cst.SubscriptElement(
        cst.Index(
            cst.Subscript(
                cst.Name("MatchIfTrue"),
                slice=(
                    cst.SubscriptElement(
                        cst.Index(
                            # MatchIfTrue takes in the original node type,
                            # and returns a boolean. So, lets convert our
                            # quoted classes (forward refs to other
                            # matchers) back to the CSTNode they refer to.
                            # We can do this because there's always a 1:1
                            # name mapping.
                            _convert_match_nodes_to_cst_nodes(oldtype)
                        ),
                    ),
                ),
            )
        )
    )


def _add_generic(name: str, oldtype: cst.BaseExpression) -> cst.BaseExpression:
    return cst.Subscript(cst.Name(name), (cst.SubscriptElement(cst.Index(oldtype)),))


class AddLogicMatchersToUnions(cst.CSTTransformer):
    def leave_Subscript(
        self, original_node: cst.Subscript, updated_node: cst.Subscript
    ) -> cst.Subscript:
        if updated_node.value.deep_equals(cst.Name("Union")):
            # Take the original node, remove do not care so we have concrete types.
            # Explicitly taking the original node because we want to discard nested
            # changes.
            concrete_only_expr = _remove_types(updated_node, ["DoNotCareSentinel"])
            return updated_node.with_changes(
                slice=[
                    *updated_node.slice,
                    cst.SubscriptElement(
                        cst.Index(_add_generic("OneOf", concrete_only_expr))
                    ),
                    cst.SubscriptElement(
                        cst.Index(_add_generic("AllOf", concrete_only_expr))
                    ),
                ]
            )
        return updated_node


class AddWildcardsToSequenceUnions(cst.CSTTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.in_match_if_true: Set[cst.CSTNode] = set()
        self.fixup_nodes: Set[cst.Subscript] = set()

    def visit_Subscript(self, node: cst.Subscript) -> None:
        # If the current node is a MatchIfTrue, we don't want to modify it.
        if node.value.deep_equals(cst.Name("MatchIfTrue")):
            self.in_match_if_true.add(node)
        # If the direct descendant is a union, lets add it to be fixed up.
        elif node.value.deep_equals(cst.Name("Sequence")):
            if self.in_match_if_true:
                # We don't want to add AtLeastN/AtMostN inside MatchIfTrue
                # type blocks, even for sequence types.
                return
            if len(node.slice) != 1:
                raise Exception(
                    "Unexpected number of sequence elements inside Sequence type "
                    + "annotation!"
                )
            nodeslice = node.slice[0].slice
            if isinstance(nodeslice, cst.Index):
                possibleunion = nodeslice.value
                if isinstance(possibleunion, cst.Subscript):
                    if possibleunion.value.deep_equals(cst.Name("Union")):
                        self.fixup_nodes.add(possibleunion)

    def leave_Subscript(
        self, original_node: cst.Subscript, updated_node: cst.Subscript
    ) -> cst.Subscript:
        if original_node in self.in_match_if_true:
            self.in_match_if_true.remove(original_node)
        if original_node in self.fixup_nodes:
            self.fixup_nodes.remove(original_node)
            return updated_node.with_changes(
                slice=[
                    *updated_node.slice,
                    cst.SubscriptElement(
                        cst.Index(_add_generic("AtLeastN", original_node))
                    ),
                    cst.SubscriptElement(
                        cst.Index(_add_generic("AtMostN", original_node))
                    ),
                ]
            )
        return updated_node


def _get_do_not_care() -> cst.SubscriptElement:
    """
    Construct a DoNotCareSentinel entry appropriate for going into a Union.
    """

    return cst.SubscriptElement(cst.Index(cst.Name("DoNotCareSentinel")))


def _get_match_metadata() -> cst.SubscriptElement:
    """
    Construct a MetadataMatchType entry appropriate for going into a Union.
    """

    return cst.SubscriptElement(cst.Index(cst.Name("MetadataMatchType")))


def _get_wrapped_union_type(
    node: cst.BaseExpression,
    addition: cst.SubscriptElement,
    *additions: cst.SubscriptElement,
) -> cst.Subscript:
    """
    Take two or more nodes, wrap them in a union type. Function signature is
    explicitly defined as taking at least one addition for type safety.

    """

    return cst.Subscript(
        cst.Name("Union"), [cst.SubscriptElement(cst.Index(node)), addition, *additions]
    )


# List of global aliases we've already generated, so we don't redefine types
_global_aliases: Set[str] = set()


@dataclass(frozen=True)
class Alias:
    name: str
    type: str


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    aliases: List[Alias]


def _get_raw_name(node: cst.CSTNode) -> Optional[str]:
    if isinstance(node, cst.Name):
        return node.value
    elif isinstance(node, cst.SimpleString):
        evaluated_value = node.evaluated_value
        if isinstance(evaluated_value, str):
            return evaluated_value
    elif isinstance(node, cst.SubscriptElement):
        return _get_raw_name(node.slice)
    elif isinstance(node, cst.Index):
        return _get_raw_name(node.value)
    else:
        return None


def _get_alias_name(node: cst.CSTNode) -> Optional[str]:
    if isinstance(node, (cst.Name, cst.SimpleString)):
        return f"{_get_raw_name(node)}MatchType"
    elif isinstance(node, cst.Subscript):
        if node.value.deep_equals(cst.Name("Union")):
            names = [_get_raw_name(s) for s in node.slice]
            if any(n is None for n in names):
                return None
            return "Or".join(n for n in names if n is not None) + "MatchType"

    return None


def _wrap_clean_type(
    aliases: List[Alias], name: Optional[str], value: cst.Subscript
) -> cst.BaseExpression:
    if name is not None:
        # We created an alias, lets use that, wrapping the alias in a do not care.
        aliases.append(Alias(name=name, type=cst.Module(body=()).code_for_node(value)))
        return _get_wrapped_union_type(cst.Name(name), _get_do_not_care())
    else:
        # Couldn't name the alias, fall back to regular node creation, add do not
        # care to the resulting type we widened.
        return value.with_changes(slice=[*value.slice, _get_do_not_care()])


def _get_clean_type_from_expression(
    aliases: List[Alias], typecst: cst.BaseExpression
) -> cst.BaseExpression:
    name = _get_alias_name(typecst)
    value = _get_wrapped_union_type(
        typecst, _get_match_metadata(), _get_match_if_true(typecst)
    )
    return _wrap_clean_type(aliases, name, value)


def _maybe_fix_sequence_in_union(
    aliases: List[Alias], typecst: cst.SubscriptElement
) -> cst.SubscriptElement:
    slc = typecst.slice
    if isinstance(slc, cst.Index):
        val = slc.value
        if isinstance(val, cst.Subscript):
            return cst.ensure_type(
                typecst.deep_replace(val, _get_clean_type_from_subscript(aliases, val)),
                cst.SubscriptElement,
            )
    return typecst


def _get_clean_type_from_union(
    aliases: List[Alias], typecst: cst.Subscript
) -> cst.BaseExpression:
    name = _get_alias_name(typecst)
    value = typecst.with_changes(
        slice=[
            *[_maybe_fix_sequence_in_union(aliases, slc) for slc in typecst.slice],
            _get_match_metadata(),
            _get_match_if_true(typecst),
        ]
    )
    return _wrap_clean_type(aliases, name, value)


def _get_clean_type_from_subscript(
    aliases: List[Alias], typecst: cst.Subscript
) -> cst.BaseExpression:
    if typecst.value.deep_equals(cst.Name("Sequence")):
        # Lets attempt to widen the sequence type and alias it.
        if len(typecst.slice) != 1:
            raise Exception("Logic error, Sequence shouldn't have more than one param!")
        inner_type = typecst.slice[0].slice
        if not isinstance(inner_type, cst.Index):
            raise Exception("Logic error, expecting Index for only Sequence element!")
        inner_type = inner_type.value

        if isinstance(inner_type, cst.Subscript):
            clean_inner_type = _get_clean_type_from_subscript(aliases, inner_type)
        elif isinstance(inner_type, (cst.Name, cst.SimpleString)):
            clean_inner_type = _get_clean_type_from_expression(aliases, inner_type)
        else:
            raise Exception("Logic error, unexpected type in Sequence!")

        return _get_wrapped_union_type(
            typecst.deep_replace(inner_type, clean_inner_type),
            _get_do_not_care(),
            _get_match_if_true(typecst),
        )
    # We can modify this as-is to add our extra values
    elif typecst.value.deep_equals(cst.Name("Union")):
        return _get_clean_type_from_union(aliases, typecst)
    else:
        # Don't handle other types like "Literal", just widen them.
        return _get_clean_type_from_expression(aliases, typecst)


def _get_clean_type_and_aliases(
    typeobj: object,
) -> Tuple[str, List[Alias]]:  # noqa: C901
    """
    Given a type object as returned by dataclasses, sanitize it and convert it
    to a type string that is appropriate for our codegen below.
    """

    # First, get the type as a parseable expression.
    typestr = repr(typeobj)
    typestr = re.sub(CLASS_RE, r"\1", typestr)
    typestr = re.sub(OPTIONAL_RE, r"typing.Optional[\1]", typestr)

    # Now, parse the expression with LibCST.
    cleanser = CleanseFullTypeNames()
    typecst = parse_expression(typestr)
    typecst = typecst.visit(cleanser)
    aliases: List[Alias] = []

    # Now, convert the type to allow for MetadataMatchType and MatchIfTrue values.
    if isinstance(typecst, cst.Subscript):
        clean_type = _get_clean_type_from_subscript(aliases, typecst)
    elif isinstance(typecst, (cst.Name, cst.SimpleString)):
        clean_type = _get_clean_type_from_expression(aliases, typecst)
    else:
        raise Exception("Logic error, unexpected top level type!")

    # Now, insert OneOf/AllOf and MatchIfTrue into unions so we can typecheck their usage.
    # This allows us to put OneOf[SomeType] or MatchIfTrue[cst.SomeType] into any
    # spot that we would have originally allowed a SomeType.
    clean_type = ensure_type(clean_type.visit(AddLogicMatchersToUnions()), cst.CSTNode)
    # Now, insert AtMostN and AtLeastN into sequence unions, so we can typecheck
    # them. This relies on the previous OneOf/AllOf insertion to ensure that all
    # sequences we care about are Sequence[Union[<x>]].
    clean_type = ensure_type(
        clean_type.visit(AddWildcardsToSequenceUnions()), cst.CSTNode
    )
    # Finally, generate the code given a default Module so we can spit it out.
    return cst.Module(body=()).code_for_node(clean_type), aliases


def _get_fields(node: Type[cst.CSTNode]) -> Generator[Field, None, None]:
    """
    Given a CSTNode, generate a field name and type string for each.
    """

    for field in fields(node) or []:
        if field.name == "_metadata":
            continue

        fieldtype, aliases = _get_clean_type_and_aliases(field.type)
        yield Field(
            name=field.name,
            type=fieldtype,
            aliases=[a for a in aliases if a.name not in _global_aliases],
        )
        _global_aliases.update(a.name for a in aliases)


def _generate_file_header() -> List[str]:
    generated_code: List[str] = []
    generated_code.append("# Copyright (c) Meta Platforms, Inc. and affiliates.")
    generated_code.append("#")
    generated_code.append(
        "# This source code is licensed under the MIT license found in the"
    )
    generated_code.append("# LICENSE file in the root directory of this source tree.")
    generated_code.append("")
    generated_code.append("")
    generated_code.append(
        "# This file was generated by libcst.codegen.gen_matcher_classes"
    )
    return generated_code


def generate_init():
    all_exports: Set[str] = set()
    generated_code: List[str] = []

    generated_code += _generate_file_header()

    all_exports.update(
        [
            "BaseMatcherNode",
            "DoNotCareSentinel",
            "DoNotCare",
            "OneOf",
            "AllOf",
            "DoesNotMatch",
            "MatchIfTrue",
            "MatchRegex",
            "MatchMetadata",
            "MatchMetadataIfTrue",
            "TypeOf",
            "ZeroOrMore",
            "AtLeastN",
            "ZeroOrOne",
            "AtMostN",
            "SaveMatchedNode",
            "extract",
            "extractall",
            "findall",
            "matches",
            "replace",
        ]
    )
    generated_code.append(
        "from libcst.matchers._decorators import call_if_inside, call_if_not_inside, visit, leave"
    )
    generated_code.append(
        "from libcst.matchers._matcher_base import BaseMatcherNode, DoNotCareSentinel, DoNotCare, OneOf, AllOf, DoesNotMatch, MatchIfTrue, MatchRegex, MatchMetadata, MatchMetadataIfTrue, TypeOf, ZeroOrMore, AtLeastN, ZeroOrOne, AtMostN, SaveMatchedNode, extract, extractall, findall, matches, replace"
    )
    all_exports.update(["call_if_inside", "call_if_not_inside", "visit", "leave"])
    generated_code.append(
        "from libcst.matchers._visitors import MatchDecoratorMismatch, MatcherDecoratableTransformer, MatcherDecoratableVisitor"
    )
    all_exports.update(
        [
            "MatchDecoratorMismatch",
            "MatcherDecoratableTransformer",
            "MatcherDecoratableVisitor",
        ]
    )

    for base in typeclasses:
        all_exports.add(base.__name__)

    for node in all_libcst_nodes:
        if node.__name__.startswith("Base"):
            continue

        kebab_node_name = PASCAL_TO_KEBAB_RE.sub("_", node.__name__).lower()
        generated_code.append(
            f"from libcst.matchers.nodes._{kebab_node_name} import {node.__name__}"
        )
        all_exports.add(node.__name__)

    # Make sure to add an __all__ for flake8 and compatibility with "from libcst.matchers import *"
    generated_code.append(f"__all__ = {repr(sorted(all_exports))}")

    return generated_code


def generate_base():
    generated_code: List[str] = []
    generated_code += _generate_file_header()
    generated_code.append(
        "from libcst.matchers._matcher_base import AbstractBaseMatcherNodeMeta"
    )

    generated_code.append("class _NodeABC(metaclass=AbstractBaseMatcherNodeMeta):")
    generated_code.append("    __slots__ = ()")

    for base in typeclasses:
        generated_code.append("")
        generated_code.append("")
        generated_code.append(f"class {base.__name__}(_NodeABC):")
        generated_code.append("    pass")

    return generated_code


def generate_match_types():
    generated_code: List[str] = []
    generated_code += _generate_file_header()
    generated_code.append("from typing import Union, TYPE_CHECKING")
    generated_code.append("")
    generated_code.append("import libcst as cst")
    generated_code.append(
        "from libcst.matchers._matcher_base import MatchIfTrue, MatchMetadata, MatchMetadataIfTrue"
    )

    imports_from_base: Set[str] = set()
    generated_imports: Set[str] = set()
    generated_aliases: List[str] = []
    for node in all_libcst_nodes:
        if node.__name__.startswith("Base"):
            continue

        has_aliases = False
        node_fields = list(_get_fields(node))
        for field in node_fields:
            for alias in field.aliases:
                # Output a separator if we're going to output any aliases
                if not has_aliases:
                    generated_aliases.append("")
                    generated_aliases.append("")
                    has_aliases = True

                # Must generate code for aliases before the class they are referenced in
                generated_aliases.append(f"{alias.name} = {alias.type}")

                nodes_to_import = re.findall(SIMPLE_STRING_RE, alias.type)

                for node_to_import in nodes_to_import:
                    node_name = node_to_import.strip('"').strip("'")
                    if node_name.startswith("Base"):
                        imports_from_base.add(node_name)
                    else:
                        kebab_node_name = PASCAL_TO_KEBAB_RE.sub("_", node_name).lower()
                        generated_imports.add(
                            f"    from libcst.matchers.nodes._{kebab_node_name} import {node_name}"
                        )

    generated_code.append(
        f'from libcst.matchers._base import {",".join(imports_from_base)}'
    )
    if len(generated_imports) > 0:
        generated_code.append("if TYPE_CHECKING:")
        generated_code += generated_imports

    generated_code.append("")
    generated_code.append(
        "MetadataMatchType = Union[MatchMetadata, MatchMetadataIfTrue]"
    )
    generated_code.append("")
    generated_code += generated_aliases

    return generated_code


class ExternalNameCollector(cst.CSTVisitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.current_node_name: Optional[str] = None
        self.imports_by_modules: Dict[str, Set[str]] = {
            "libcst.matchers._match_types": set(["MetadataMatchType"]),
            "libcst.matchers._matcher_base": set(
                ["BaseMatcherNode", "DoNotCare", "DoNotCareSentinel", "OneOf", "AllOf"]
            ),
            "typing": set(["Union"]),
        }

    def visit_Attribute(self, attr: cst.Attribute) -> bool:
        # ignore attributes like 'cst.<member>'
        return False

    def visit_Name(self, name: cst.Name) -> bool:
        self.handle_type_identifier(name.value)
        return False

    def visit_SimpleString(self, string: cst.SimpleString) -> bool:
        self.handle_type_identifier(string.evaluated_value)
        return False

    def handle_type_identifier(self, identifier: str) -> None:
        if identifier.endswith("MatchType"):
            self.add_import_of_module(identifier, "libcst.matchers._match_types")
        elif identifier in [
            "MatchIfTrue",
            "BaseMatcherNode",
            "DoNotCare",
            "DoNotCareSentinel",
            "OneOf",
            "AllOf",
            "AtLeastN",
            "AtMostN",
        ]:
            self.add_import_of_module(identifier, "libcst.matchers._matcher_base")
        elif identifier.startswith("Base"):
            self.add_import_of_module(identifier, "libcst.matchers._base")
        elif identifier in ["Literal", "Optional", "Sequence", "Union"]:
            self.add_import_of_module(identifier, "typing")
        else:
            # assume it is a node
            # - ignore stars etc
            # - str is built-in name, not libcst node
            # - current node exists already, no need to import
            if (
                identifier.isalpha()
                and identifier != "str"
                and identifier != self.current_node_name
            ):
                kebab_node_name = PASCAL_TO_KEBAB_RE.sub("_", identifier).lower()
                self.add_import_of_module(
                    identifier, f"libcst.matchers.nodes._{kebab_node_name}"
                )

    def add_import_of_module(self, name_to_import: str, module: str) -> None:
        if module not in self.imports_by_modules:
            self.imports_by_modules[module] = set()
        self.imports_by_modules[module].add(name_to_import)


def generate_nodes_matchers():
    # tuples with node name in kebab case and file content
    nodes_matchers: List[Tuple[str, str]] = []

    for node in all_libcst_nodes:
        if node.__name__.startswith("Base"):
            continue

        generated_code: List[str] = []
        generated_class_code: List[str] = []
        classes: List[str] = []
        base_members_to_import: Set[str] = set()
        external_name_collector = ExternalNameCollector()
        external_name_collector.current_node_name = node.__name__

        for tc in typeclasses:
            if issubclass(node, tc):
                external_name_collector.handle_type_identifier(tc.__name__)
                classes.append(tc.__name__)
                base_members_to_import.add(tc.__name__)
        classes.append("BaseMatcherNode")

        node_fields = list(_get_fields(node))

        generated_code += _generate_file_header()
        generated_code.append("from dataclasses import dataclass")
        generated_code.append("")
        generated_code.append("import libcst as cst")
        generated_class_code.append("")
        generated_class_code.append("")
        generated_class_code.append(
            "@dataclass(frozen=True, eq=False, unsafe_hash=False)"
        )
        generated_class_code.append(f'class {node.__name__}({", ".join(classes)}):')

        for field in node_fields:
            generated_class_code.append(f"    {field.name}: {field.type} = DoNotCare()")
            cst.parse_expression(field.type).visit(external_name_collector)

        # Add special metadata field
        generated_class_code.append(
            "    metadata: Union[MetadataMatchType, DoNotCareSentinel, OneOf[MetadataMatchType], AllOf[MetadataMatchType]] = DoNotCare()"
        )

        kebab_node_name = PASCAL_TO_KEBAB_RE.sub("_", node.__name__).lower()

        for module, imports_set in external_name_collector.imports_by_modules.items():
            names_to_import = ",".join(imports_set)
            if len(names_to_import) > 0:
                generated_code.append(f"from {module} import {names_to_import}")

        generated_code += generated_class_code
        nodes_matchers.append((kebab_node_name, generated_code))

    return nodes_matchers
