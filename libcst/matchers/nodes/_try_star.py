# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file was generated by libcst.codegen.gen_matcher_classes
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import libcst as cst
from libcst.matchers._base import BaseCompoundStatement, BaseStatement
from libcst.matchers._match_types import (
    BaseSuiteMatchType,
    EmptyLineMatchType,
    ExceptStarHandlerMatchType,
    MetadataMatchType,
    SimpleWhitespaceMatchType,
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
from libcst.matchers.nodes._else import Else
from libcst.matchers.nodes._finally import Finally


@dataclass(frozen=True, eq=False, unsafe_hash=False)
class TryStar(BaseCompoundStatement, BaseStatement, BaseMatcherNode):
    body: Union[
        BaseSuiteMatchType,
        DoNotCareSentinel,
        OneOf[BaseSuiteMatchType],
        AllOf[BaseSuiteMatchType],
    ] = DoNotCare()
    handlers: Union[
        Sequence[
            Union[
                ExceptStarHandlerMatchType,
                DoNotCareSentinel,
                OneOf[ExceptStarHandlerMatchType],
                AllOf[ExceptStarHandlerMatchType],
                AtLeastN[
                    Union[
                        ExceptStarHandlerMatchType,
                        DoNotCareSentinel,
                        OneOf[ExceptStarHandlerMatchType],
                        AllOf[ExceptStarHandlerMatchType],
                    ]
                ],
                AtMostN[
                    Union[
                        ExceptStarHandlerMatchType,
                        DoNotCareSentinel,
                        OneOf[ExceptStarHandlerMatchType],
                        AllOf[ExceptStarHandlerMatchType],
                    ]
                ],
            ]
        ],
        DoNotCareSentinel,
        MatchIfTrue[Sequence[cst.ExceptStarHandler]],
        OneOf[
            Union[
                Sequence[
                    Union[
                        ExceptStarHandlerMatchType,
                        OneOf[ExceptStarHandlerMatchType],
                        AllOf[ExceptStarHandlerMatchType],
                        AtLeastN[
                            Union[
                                ExceptStarHandlerMatchType,
                                OneOf[ExceptStarHandlerMatchType],
                                AllOf[ExceptStarHandlerMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                ExceptStarHandlerMatchType,
                                OneOf[ExceptStarHandlerMatchType],
                                AllOf[ExceptStarHandlerMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.ExceptStarHandler]],
            ]
        ],
        AllOf[
            Union[
                Sequence[
                    Union[
                        ExceptStarHandlerMatchType,
                        OneOf[ExceptStarHandlerMatchType],
                        AllOf[ExceptStarHandlerMatchType],
                        AtLeastN[
                            Union[
                                ExceptStarHandlerMatchType,
                                OneOf[ExceptStarHandlerMatchType],
                                AllOf[ExceptStarHandlerMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                ExceptStarHandlerMatchType,
                                OneOf[ExceptStarHandlerMatchType],
                                AllOf[ExceptStarHandlerMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.ExceptStarHandler]],
            ]
        ],
    ] = DoNotCare()
    orelse: Union[
        Optional["Else"],
        MetadataMatchType,
        MatchIfTrue[Optional[cst.Else]],
        DoNotCareSentinel,
        OneOf[
            Union[Optional["Else"], MetadataMatchType, MatchIfTrue[Optional[cst.Else]]]
        ],
        AllOf[
            Union[Optional["Else"], MetadataMatchType, MatchIfTrue[Optional[cst.Else]]]
        ],
    ] = DoNotCare()
    finalbody: Union[
        Optional["Finally"],
        MetadataMatchType,
        MatchIfTrue[Optional[cst.Finally]],
        DoNotCareSentinel,
        OneOf[
            Union[
                Optional["Finally"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.Finally]],
            ]
        ],
        AllOf[
            Union[
                Optional["Finally"],
                MetadataMatchType,
                MatchIfTrue[Optional[cst.Finally]],
            ]
        ],
    ] = DoNotCare()
    leading_lines: Union[
        Sequence[
            Union[
                EmptyLineMatchType,
                DoNotCareSentinel,
                OneOf[EmptyLineMatchType],
                AllOf[EmptyLineMatchType],
                AtLeastN[
                    Union[
                        EmptyLineMatchType,
                        DoNotCareSentinel,
                        OneOf[EmptyLineMatchType],
                        AllOf[EmptyLineMatchType],
                    ]
                ],
                AtMostN[
                    Union[
                        EmptyLineMatchType,
                        DoNotCareSentinel,
                        OneOf[EmptyLineMatchType],
                        AllOf[EmptyLineMatchType],
                    ]
                ],
            ]
        ],
        DoNotCareSentinel,
        MatchIfTrue[Sequence[cst.EmptyLine]],
        OneOf[
            Union[
                Sequence[
                    Union[
                        EmptyLineMatchType,
                        OneOf[EmptyLineMatchType],
                        AllOf[EmptyLineMatchType],
                        AtLeastN[
                            Union[
                                EmptyLineMatchType,
                                OneOf[EmptyLineMatchType],
                                AllOf[EmptyLineMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                EmptyLineMatchType,
                                OneOf[EmptyLineMatchType],
                                AllOf[EmptyLineMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.EmptyLine]],
            ]
        ],
        AllOf[
            Union[
                Sequence[
                    Union[
                        EmptyLineMatchType,
                        OneOf[EmptyLineMatchType],
                        AllOf[EmptyLineMatchType],
                        AtLeastN[
                            Union[
                                EmptyLineMatchType,
                                OneOf[EmptyLineMatchType],
                                AllOf[EmptyLineMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                EmptyLineMatchType,
                                OneOf[EmptyLineMatchType],
                                AllOf[EmptyLineMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.EmptyLine]],
            ]
        ],
    ] = DoNotCare()
    whitespace_before_colon: Union[
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
