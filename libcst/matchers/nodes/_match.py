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
    BaseExpressionMatchType,
    EmptyLineMatchType,
    MatchCaseMatchType,
    MetadataMatchType,
    SimpleWhitespaceMatchType,
    TrailingWhitespaceMatchType,
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


@dataclass(frozen=True, eq=False, unsafe_hash=False)
class Match(BaseCompoundStatement, BaseStatement, BaseMatcherNode):
    subject: Union[
        BaseExpressionMatchType,
        DoNotCareSentinel,
        OneOf[BaseExpressionMatchType],
        AllOf[BaseExpressionMatchType],
    ] = DoNotCare()
    cases: Union[
        Sequence[
            Union[
                MatchCaseMatchType,
                DoNotCareSentinel,
                OneOf[MatchCaseMatchType],
                AllOf[MatchCaseMatchType],
                AtLeastN[
                    Union[
                        MatchCaseMatchType,
                        DoNotCareSentinel,
                        OneOf[MatchCaseMatchType],
                        AllOf[MatchCaseMatchType],
                    ]
                ],
                AtMostN[
                    Union[
                        MatchCaseMatchType,
                        DoNotCareSentinel,
                        OneOf[MatchCaseMatchType],
                        AllOf[MatchCaseMatchType],
                    ]
                ],
            ]
        ],
        DoNotCareSentinel,
        MatchIfTrue[Sequence[cst.MatchCase]],
        OneOf[
            Union[
                Sequence[
                    Union[
                        MatchCaseMatchType,
                        OneOf[MatchCaseMatchType],
                        AllOf[MatchCaseMatchType],
                        AtLeastN[
                            Union[
                                MatchCaseMatchType,
                                OneOf[MatchCaseMatchType],
                                AllOf[MatchCaseMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                MatchCaseMatchType,
                                OneOf[MatchCaseMatchType],
                                AllOf[MatchCaseMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.MatchCase]],
            ]
        ],
        AllOf[
            Union[
                Sequence[
                    Union[
                        MatchCaseMatchType,
                        OneOf[MatchCaseMatchType],
                        AllOf[MatchCaseMatchType],
                        AtLeastN[
                            Union[
                                MatchCaseMatchType,
                                OneOf[MatchCaseMatchType],
                                AllOf[MatchCaseMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                MatchCaseMatchType,
                                OneOf[MatchCaseMatchType],
                                AllOf[MatchCaseMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.MatchCase]],
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
    whitespace_after_match: Union[
        SimpleWhitespaceMatchType,
        DoNotCareSentinel,
        OneOf[SimpleWhitespaceMatchType],
        AllOf[SimpleWhitespaceMatchType],
    ] = DoNotCare()
    whitespace_before_colon: Union[
        SimpleWhitespaceMatchType,
        DoNotCareSentinel,
        OneOf[SimpleWhitespaceMatchType],
        AllOf[SimpleWhitespaceMatchType],
    ] = DoNotCare()
    whitespace_after_colon: Union[
        TrailingWhitespaceMatchType,
        DoNotCareSentinel,
        OneOf[TrailingWhitespaceMatchType],
        AllOf[TrailingWhitespaceMatchType],
    ] = DoNotCare()
    indent: Union[
        Optional[str],
        MetadataMatchType,
        MatchIfTrue[Optional[str]],
        DoNotCareSentinel,
        OneOf[Union[Optional[str], MetadataMatchType, MatchIfTrue[Optional[str]]]],
        AllOf[Union[Optional[str], MetadataMatchType, MatchIfTrue[Optional[str]]]],
    ] = DoNotCare()
    footer: Union[
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
    metadata: Union[
        MetadataMatchType,
        DoNotCareSentinel,
        OneOf[MetadataMatchType],
        AllOf[MetadataMatchType],
    ] = DoNotCare()
