# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file was generated by libcst.codegen.gen_matcher_classes
from dataclasses import dataclass
from typing import Optional, Union

import libcst as cst
from libcst.matchers._base import BaseExpression, BaseSmallStatement
from libcst.matchers._match_types import (
    BaseExpressionMatchType,
    CommaMatchType,
    MetadataMatchType,
    SemicolonMatchType,
    SimpleWhitespaceMatchType,
)
from libcst.matchers._matcher_base import (
    AllOf,
    BaseMatcherNode,
    DoNotCare,
    DoNotCareSentinel,
    MatchIfTrue,
    OneOf,
)


@dataclass(frozen=True, eq=False, unsafe_hash=False)
class Assert(BaseSmallStatement, BaseMatcherNode):
    test: Union[
        BaseExpressionMatchType,
        DoNotCareSentinel,
        OneOf[BaseExpressionMatchType],
        AllOf[BaseExpressionMatchType],
    ] = DoNotCare()
    msg: Union[
        Optional["BaseExpression"],
        MetadataMatchType,
        MatchIfTrue[Optional[cst.BaseExpression]],
        DoNotCareSentinel,
        OneOf[
            Union[
                Optional["BaseExpression"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.BaseExpression]],
            ]
        ],
        AllOf[
            Union[
                Optional["BaseExpression"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.BaseExpression]],
            ]
        ],
    ] = DoNotCare()
    comma: Union[
        CommaMatchType, DoNotCareSentinel, OneOf[CommaMatchType], AllOf[CommaMatchType]
    ] = DoNotCare()
    whitespace_after_assert: Union[
        SimpleWhitespaceMatchType,
        DoNotCareSentinel,
        OneOf[SimpleWhitespaceMatchType],
        AllOf[SimpleWhitespaceMatchType],
    ] = DoNotCare()
    semicolon: Union[
        SemicolonMatchType,
        DoNotCareSentinel,
        OneOf[SemicolonMatchType],
        AllOf[SemicolonMatchType],
    ] = DoNotCare()
    metadata: Union[
        MetadataMatchType,
        DoNotCareSentinel,
        OneOf[MetadataMatchType],
        AllOf[MetadataMatchType],
    ] = DoNotCare()
