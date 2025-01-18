# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file was generated by libcst.codegen.gen_matcher_classes
from dataclasses import dataclass
from typing import Sequence, Union

import libcst as cst
from libcst.matchers._base import BaseExpression, BaseSet
from libcst.matchers._match_types import (
    BaseElementMatchType,
    LeftCurlyBraceMatchType,
    LeftParenMatchType,
    MetadataMatchType,
    RightCurlyBraceMatchType,
    RightParenMatchType,
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
class Set(BaseExpression, BaseSet, BaseMatcherNode):
    elements: Union[
        Sequence[
            Union[
                BaseElementMatchType,
                DoNotCareSentinel,
                OneOf[BaseElementMatchType],
                AllOf[BaseElementMatchType],
                AtLeastN[
                    Union[
                        BaseElementMatchType,
                        DoNotCareSentinel,
                        OneOf[BaseElementMatchType],
                        AllOf[BaseElementMatchType],
                    ]
                ],
                AtMostN[
                    Union[
                        BaseElementMatchType,
                        DoNotCareSentinel,
                        OneOf[BaseElementMatchType],
                        AllOf[BaseElementMatchType],
                    ]
                ],
            ]
        ],
        DoNotCareSentinel,
        MatchIfTrue[Sequence[cst.BaseElement]],
        OneOf[
            Union[
                Sequence[
                    Union[
                        BaseElementMatchType,
                        OneOf[BaseElementMatchType],
                        AllOf[BaseElementMatchType],
                        AtLeastN[
                            Union[
                                BaseElementMatchType,
                                OneOf[BaseElementMatchType],
                                AllOf[BaseElementMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                BaseElementMatchType,
                                OneOf[BaseElementMatchType],
                                AllOf[BaseElementMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.BaseElement]],
            ]
        ],
        AllOf[
            Union[
                Sequence[
                    Union[
                        BaseElementMatchType,
                        OneOf[BaseElementMatchType],
                        AllOf[BaseElementMatchType],
                        AtLeastN[
                            Union[
                                BaseElementMatchType,
                                OneOf[BaseElementMatchType],
                                AllOf[BaseElementMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                BaseElementMatchType,
                                OneOf[BaseElementMatchType],
                                AllOf[BaseElementMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.BaseElement]],
            ]
        ],
    ] = DoNotCare()
    lbrace: Union[
        LeftCurlyBraceMatchType,
        DoNotCareSentinel,
        OneOf[LeftCurlyBraceMatchType],
        AllOf[LeftCurlyBraceMatchType],
    ] = DoNotCare()
    rbrace: Union[
        RightCurlyBraceMatchType,
        DoNotCareSentinel,
        OneOf[RightCurlyBraceMatchType],
        AllOf[RightCurlyBraceMatchType],
    ] = DoNotCare()
    lpar: Union[
        Sequence[
            Union[
                LeftParenMatchType,
                DoNotCareSentinel,
                OneOf[LeftParenMatchType],
                AllOf[LeftParenMatchType],
                AtLeastN[
                    Union[
                        LeftParenMatchType,
                        DoNotCareSentinel,
                        OneOf[LeftParenMatchType],
                        AllOf[LeftParenMatchType],
                    ]
                ],
                AtMostN[
                    Union[
                        LeftParenMatchType,
                        DoNotCareSentinel,
                        OneOf[LeftParenMatchType],
                        AllOf[LeftParenMatchType],
                    ]
                ],
            ]
        ],
        DoNotCareSentinel,
        MatchIfTrue[Sequence[cst.LeftParen]],
        OneOf[
            Union[
                Sequence[
                    Union[
                        LeftParenMatchType,
                        OneOf[LeftParenMatchType],
                        AllOf[LeftParenMatchType],
                        AtLeastN[
                            Union[
                                LeftParenMatchType,
                                OneOf[LeftParenMatchType],
                                AllOf[LeftParenMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                LeftParenMatchType,
                                OneOf[LeftParenMatchType],
                                AllOf[LeftParenMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.LeftParen]],
            ]
        ],
        AllOf[
            Union[
                Sequence[
                    Union[
                        LeftParenMatchType,
                        OneOf[LeftParenMatchType],
                        AllOf[LeftParenMatchType],
                        AtLeastN[
                            Union[
                                LeftParenMatchType,
                                OneOf[LeftParenMatchType],
                                AllOf[LeftParenMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                LeftParenMatchType,
                                OneOf[LeftParenMatchType],
                                AllOf[LeftParenMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.LeftParen]],
            ]
        ],
    ] = DoNotCare()
    rpar: Union[
        Sequence[
            Union[
                RightParenMatchType,
                DoNotCareSentinel,
                OneOf[RightParenMatchType],
                AllOf[RightParenMatchType],
                AtLeastN[
                    Union[
                        RightParenMatchType,
                        DoNotCareSentinel,
                        OneOf[RightParenMatchType],
                        AllOf[RightParenMatchType],
                    ]
                ],
                AtMostN[
                    Union[
                        RightParenMatchType,
                        DoNotCareSentinel,
                        OneOf[RightParenMatchType],
                        AllOf[RightParenMatchType],
                    ]
                ],
            ]
        ],
        DoNotCareSentinel,
        MatchIfTrue[Sequence[cst.RightParen]],
        OneOf[
            Union[
                Sequence[
                    Union[
                        RightParenMatchType,
                        OneOf[RightParenMatchType],
                        AllOf[RightParenMatchType],
                        AtLeastN[
                            Union[
                                RightParenMatchType,
                                OneOf[RightParenMatchType],
                                AllOf[RightParenMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                RightParenMatchType,
                                OneOf[RightParenMatchType],
                                AllOf[RightParenMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.RightParen]],
            ]
        ],
        AllOf[
            Union[
                Sequence[
                    Union[
                        RightParenMatchType,
                        OneOf[RightParenMatchType],
                        AllOf[RightParenMatchType],
                        AtLeastN[
                            Union[
                                RightParenMatchType,
                                OneOf[RightParenMatchType],
                                AllOf[RightParenMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                RightParenMatchType,
                                OneOf[RightParenMatchType],
                                AllOf[RightParenMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.RightParen]],
            ]
        ],
    ] = DoNotCare()
    metadata: Union[
        MetadataMatchType,
        DoNotCareSentinel,
        OneOf[MetadataMatchType],
        AllOf[MetadataMatchType],
    ] = DoNotCare()
