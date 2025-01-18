# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file was generated by libcst.codegen.gen_matcher_classes
from dataclasses import dataclass
from typing import Optional, Union

import libcst as cst
from libcst.matchers._base import BaseExpression
from libcst.matchers._match_types import (
    AssignEqualMatchType,
    BaseParenthesizableWhitespaceMatchType,
    CommaMatchType,
    MetadataMatchType,
    NameMatchType,
    strMatchType,
)
from libcst.matchers._matcher_base import (
    AllOf,
    BaseMatcherNode,
    DoNotCare,
    DoNotCareSentinel,
    MatchIfTrue,
    OneOf,
)
from libcst.matchers.nodes._annotation import Annotation


@dataclass(frozen=True, eq=False, unsafe_hash=False)
class Param(BaseMatcherNode):
    name: Union[
        NameMatchType, DoNotCareSentinel, OneOf[NameMatchType], AllOf[NameMatchType]
    ] = DoNotCare()
    annotation: Union[
        Optional["Annotation"],
        MetadataMatchType,
        MatchIfTrue[Optional[cst.Annotation]],
        DoNotCareSentinel,
        OneOf[
            Union[
                Optional["Annotation"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.Annotation]],
            ]
        ],
        AllOf[
            Union[
                Optional["Annotation"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.Annotation]],
            ]
        ],
    ] = DoNotCare()
    equal: Union[
        AssignEqualMatchType,
        DoNotCareSentinel,
        OneOf[AssignEqualMatchType],
        AllOf[AssignEqualMatchType],
    ] = DoNotCare()
    default: Union[
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
    star: Union[
        strMatchType, DoNotCareSentinel, OneOf[strMatchType], AllOf[strMatchType]
    ] = DoNotCare()
    whitespace_after_star: Union[
        BaseParenthesizableWhitespaceMatchType,
        DoNotCareSentinel,
        OneOf[BaseParenthesizableWhitespaceMatchType],
        AllOf[BaseParenthesizableWhitespaceMatchType],
    ] = DoNotCare()
    whitespace_after_param: Union[
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
