# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file was generated by libcst.codegen.gen_matcher_classes
from dataclasses import dataclass
from typing import Union

import libcst as cst
from libcst.matchers._match_types import (
    MetadataMatchType,
    NameMatchType,
    SimpleWhitespaceMatchType,
)
from libcst.matchers._matcher_base import (
    AllOf,
    BaseMatcherNode,
    DoNotCare,
    DoNotCareSentinel,
    OneOf,
)


@dataclass(frozen=True, eq=False, unsafe_hash=False)
class ParamSpec(BaseMatcherNode):
    name: Union[
        NameMatchType, DoNotCareSentinel, OneOf[NameMatchType], AllOf[NameMatchType]
    ] = DoNotCare()
    whitespace_after_star: Union[
        SimpleWhitespaceMatchType,
        DoNotCareSentinel,
        OneOf[SimpleWhitespaceMatchType],
        AllOf[SimpleWhitespaceMatchType],
    ] = DoNotCare()
    metadata: Union[
        MetadataMatchType,
        DoNotCareSentinel,
        OneOf[MetadataMatchType],
        AllOf[MetadataMatchType],
    ] = DoNotCare()
