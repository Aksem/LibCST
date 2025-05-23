# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file was generated by libcst.codegen.gen_matcher_classes
from dataclasses import dataclass
from typing import Union

import libcst as cst
from libcst.matchers._base import BaseBinaryOp
from libcst.matchers._match_types import (
    BaseParenthesizableWhitespaceMatchType,
    MetadataMatchType,
)
from libcst.matchers._matcher_base import (
    AllOf,
    BaseMatcherNode,
    DoNotCare,
    DoNotCareSentinel,
    OneOf,
)


@dataclass(frozen=True, eq=False, unsafe_hash=False)
class Add(BaseBinaryOp, BaseMatcherNode):
    whitespace_before: Union[
        BaseParenthesizableWhitespaceMatchType,
        DoNotCareSentinel,
        OneOf[BaseParenthesizableWhitespaceMatchType],
        AllOf[BaseParenthesizableWhitespaceMatchType],
    ] = DoNotCare()
    whitespace_after: Union[
        BaseParenthesizableWhitespaceMatchType,
        DoNotCareSentinel,
        OneOf[BaseParenthesizableWhitespaceMatchType],
        AllOf[BaseParenthesizableWhitespaceMatchType],
    ] = DoNotCare()
    metadata: Union[
        MetadataMatchType,
        DoNotCareSentinel,
        OneOf[MetadataMatchType],
        AllOf[MetadataMatchType],
    ] = DoNotCare()
