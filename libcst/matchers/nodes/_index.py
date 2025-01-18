# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file was generated by libcst.codegen.gen_matcher_classes
from dataclasses import dataclass
from typing import Literal, Optional, Union

import libcst as cst
from libcst.matchers._base import BaseParenthesizableWhitespace, BaseSlice
from libcst.matchers._match_types import BaseExpressionMatchType, MetadataMatchType
from libcst.matchers._matcher_base import (
    AllOf,
    BaseMatcherNode,
    DoNotCare,
    DoNotCareSentinel,
    MatchIfTrue,
    OneOf,
)


@dataclass(frozen=True, eq=False, unsafe_hash=False)
class Index(BaseSlice, BaseMatcherNode):
    value: Union[
        BaseExpressionMatchType,
        DoNotCareSentinel,
        OneOf[BaseExpressionMatchType],
        AllOf[BaseExpressionMatchType],
    ] = DoNotCare()
    star: Union[
        Optional[Literal["*"]],
        MetadataMatchType,
        MatchIfTrue[Optional[Literal["*"]]],
        DoNotCareSentinel,
        OneOf[
            Union[
                Optional[Literal["*"]],
                MetadataMatchType,
                MatchIfTrue[Optional[Literal["*"]]],
            ]
        ],
        AllOf[
            Union[
                Optional[Literal["*"]],
                MetadataMatchType,
                MatchIfTrue[Optional[Literal["*"]]],
            ]
        ],
    ] = DoNotCare()
    whitespace_after_star: Union[
        Optional["BaseParenthesizableWhitespace"],
        MetadataMatchType,
        MatchIfTrue[Optional[cst.BaseParenthesizableWhitespace]],
        DoNotCareSentinel,
        OneOf[
            Union[
                Optional["BaseParenthesizableWhitespace"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.BaseParenthesizableWhitespace]],
            ]
        ],
        AllOf[
            Union[
                Optional["BaseParenthesizableWhitespace"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.BaseParenthesizableWhitespace]],
            ]
        ],
    ] = DoNotCare()
    metadata: Union[
        MetadataMatchType,
        DoNotCareSentinel,
        OneOf[MetadataMatchType],
        AllOf[MetadataMatchType],
    ] = DoNotCare()
