# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file was generated by libcst.codegen.gen_matcher_classes
from dataclasses import dataclass
from typing import Optional, Union

import libcst as cst
from libcst.matchers._match_types import MetadataMatchType
from libcst.matchers._matcher_base import (
    AllOf,
    BaseMatcherNode,
    DoNotCare,
    DoNotCareSentinel,
    MatchIfTrue,
    OneOf,
)


@dataclass(frozen=True, eq=False, unsafe_hash=False)
class Newline(BaseMatcherNode):
    value: Union[
        Optional[str],
        MetadataMatchType,
        MatchIfTrue[Optional[str]],
        DoNotCareSentinel,
        OneOf[Union[Optional[str], MetadataMatchType, MatchIfTrue[Optional[str]]]],
        AllOf[Union[Optional[str], MetadataMatchType, MatchIfTrue[Optional[str]]]],
    ] = DoNotCare()
    metadata: Union[
        MetadataMatchType,
        DoNotCareSentinel,
        OneOf[MetadataMatchType],
        AllOf[MetadataMatchType],
    ] = DoNotCare()
