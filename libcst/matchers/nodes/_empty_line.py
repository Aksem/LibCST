# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file was generated by libcst.codegen.gen_matcher_classes
from dataclasses import dataclass
from typing import Optional, Union

import libcst as cst
from libcst.matchers._match_types import (
    boolMatchType,
    MetadataMatchType,
    NewlineMatchType,
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
from libcst.matchers.nodes._comment import Comment


@dataclass(frozen=True, eq=False, unsafe_hash=False)
class EmptyLine(BaseMatcherNode):
    indent: Union[
        boolMatchType, DoNotCareSentinel, OneOf[boolMatchType], AllOf[boolMatchType]
    ] = DoNotCare()
    whitespace: Union[
        SimpleWhitespaceMatchType,
        DoNotCareSentinel,
        OneOf[SimpleWhitespaceMatchType],
        AllOf[SimpleWhitespaceMatchType],
    ] = DoNotCare()
    comment: Union[
        Optional["Comment"],
        MetadataMatchType,
        MatchIfTrue[Optional[cst.Comment]],
        DoNotCareSentinel,
        OneOf[
            Union[
                Optional["Comment"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.Comment]],
            ]
        ],
        AllOf[
            Union[
                Optional["Comment"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.Comment]],
            ]
        ],
    ] = DoNotCare()
    newline: Union[
        NewlineMatchType,
        DoNotCareSentinel,
        OneOf[NewlineMatchType],
        AllOf[NewlineMatchType],
    ] = DoNotCare()
    metadata: Union[
        MetadataMatchType,
        DoNotCareSentinel,
        OneOf[MetadataMatchType],
        AllOf[MetadataMatchType],
    ] = DoNotCare()
