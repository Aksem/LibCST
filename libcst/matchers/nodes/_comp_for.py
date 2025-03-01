# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file was generated by libcst.codegen.gen_matcher_classes
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import libcst as cst
from libcst.matchers._match_types import (
    BaseAssignTargetExpressionMatchType,
    BaseExpressionMatchType,
    BaseParenthesizableWhitespaceMatchType,
    CompIfMatchType,
    MetadataMatchType,
)
from libcst.matchers._matcher_base import (
    AllOf,
    AtLeastN,
    AtMostN,
    BaseMatcherNode,
    DoNotCare,
    DoNotCareSentinel,
    MatchIfTrue,
    OneOf,
)
from libcst.matchers.nodes._asynchronous import Asynchronous


@dataclass(frozen=True, eq=False, unsafe_hash=False)
class CompFor(BaseMatcherNode):
    target: Union[
        BaseAssignTargetExpressionMatchType,
        DoNotCareSentinel,
        OneOf[BaseAssignTargetExpressionMatchType],
        AllOf[BaseAssignTargetExpressionMatchType],
    ] = DoNotCare()
    iter: Union[
        BaseExpressionMatchType,
        DoNotCareSentinel,
        OneOf[BaseExpressionMatchType],
        AllOf[BaseExpressionMatchType],
    ] = DoNotCare()
    ifs: Union[
        Sequence[
            Union[
                CompIfMatchType,
                DoNotCareSentinel,
                OneOf[CompIfMatchType],
                AllOf[CompIfMatchType],
                AtLeastN[
                    Union[
                        CompIfMatchType,
                        DoNotCareSentinel,
                        OneOf[CompIfMatchType],
                        AllOf[CompIfMatchType],
                    ]
                ],
                AtMostN[
                    Union[
                        CompIfMatchType,
                        DoNotCareSentinel,
                        OneOf[CompIfMatchType],
                        AllOf[CompIfMatchType],
                    ]
                ],
            ]
        ],
        DoNotCareSentinel,
        MatchIfTrue[Sequence[cst.CompIf]],
        OneOf[
            Union[
                Sequence[
                    Union[
                        CompIfMatchType,
                        OneOf[CompIfMatchType],
                        AllOf[CompIfMatchType],
                        AtLeastN[
                            Union[
                                CompIfMatchType,
                                OneOf[CompIfMatchType],
                                AllOf[CompIfMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                CompIfMatchType,
                                OneOf[CompIfMatchType],
                                AllOf[CompIfMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.CompIf]],
            ]
        ],
        AllOf[
            Union[
                Sequence[
                    Union[
                        CompIfMatchType,
                        OneOf[CompIfMatchType],
                        AllOf[CompIfMatchType],
                        AtLeastN[
                            Union[
                                CompIfMatchType,
                                OneOf[CompIfMatchType],
                                AllOf[CompIfMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                CompIfMatchType,
                                OneOf[CompIfMatchType],
                                AllOf[CompIfMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.CompIf]],
            ]
        ],
    ] = DoNotCare()
    inner_for_in: Union[
        Optional["CompFor"],
        MetadataMatchType,
        MatchIfTrue[Optional[cst.CompFor]],
        DoNotCareSentinel,
        OneOf[
            Union[
                Optional["CompFor"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.CompFor]],
            ]
        ],
        AllOf[
            Union[
                Optional["CompFor"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.CompFor]],
            ]
        ],
    ] = DoNotCare()
    asynchronous: Union[
        Optional["Asynchronous"],
        MetadataMatchType,
        MatchIfTrue[Optional[cst.Asynchronous]],
        DoNotCareSentinel,
        OneOf[
            Union[
                Optional["Asynchronous"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.Asynchronous]],
            ]
        ],
        AllOf[
            Union[
                Optional["Asynchronous"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.Asynchronous]],
            ]
        ],
    ] = DoNotCare()
    whitespace_before: Union[
        BaseParenthesizableWhitespaceMatchType,
        DoNotCareSentinel,
        OneOf[BaseParenthesizableWhitespaceMatchType],
        AllOf[BaseParenthesizableWhitespaceMatchType],
    ] = DoNotCare()
    whitespace_after_for: Union[
        BaseParenthesizableWhitespaceMatchType,
        DoNotCareSentinel,
        OneOf[BaseParenthesizableWhitespaceMatchType],
        AllOf[BaseParenthesizableWhitespaceMatchType],
    ] = DoNotCare()
    whitespace_before_in: Union[
        BaseParenthesizableWhitespaceMatchType,
        DoNotCareSentinel,
        OneOf[BaseParenthesizableWhitespaceMatchType],
        AllOf[BaseParenthesizableWhitespaceMatchType],
    ] = DoNotCare()
    whitespace_after_in: Union[
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
