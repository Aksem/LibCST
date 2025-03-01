# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file was generated by libcst.codegen.gen_matcher_classes
from dataclasses import dataclass
from typing import Sequence, Union

import libcst as cst
from libcst.matchers._match_types import (
    LeftSquareBracketMatchType,
    MetadataMatchType,
    RightSquareBracketMatchType,
    TypeParamMatchType,
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
class TypeParameters(BaseMatcherNode):
    params: Union[
        Sequence[
            Union[
                TypeParamMatchType,
                DoNotCareSentinel,
                OneOf[TypeParamMatchType],
                AllOf[TypeParamMatchType],
                AtLeastN[
                    Union[
                        TypeParamMatchType,
                        DoNotCareSentinel,
                        OneOf[TypeParamMatchType],
                        AllOf[TypeParamMatchType],
                    ]
                ],
                AtMostN[
                    Union[
                        TypeParamMatchType,
                        DoNotCareSentinel,
                        OneOf[TypeParamMatchType],
                        AllOf[TypeParamMatchType],
                    ]
                ],
            ]
        ],
        DoNotCareSentinel,
        MatchIfTrue[Sequence[cst.TypeParam]],
        OneOf[
            Union[
                Sequence[
                    Union[
                        TypeParamMatchType,
                        OneOf[TypeParamMatchType],
                        AllOf[TypeParamMatchType],
                        AtLeastN[
                            Union[
                                TypeParamMatchType,
                                OneOf[TypeParamMatchType],
                                AllOf[TypeParamMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                TypeParamMatchType,
                                OneOf[TypeParamMatchType],
                                AllOf[TypeParamMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.TypeParam]],
            ]
        ],
        AllOf[
            Union[
                Sequence[
                    Union[
                        TypeParamMatchType,
                        OneOf[TypeParamMatchType],
                        AllOf[TypeParamMatchType],
                        AtLeastN[
                            Union[
                                TypeParamMatchType,
                                OneOf[TypeParamMatchType],
                                AllOf[TypeParamMatchType],
                            ]
                        ],
                        AtMostN[
                            Union[
                                TypeParamMatchType,
                                OneOf[TypeParamMatchType],
                                AllOf[TypeParamMatchType],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[Sequence[cst.TypeParam]],
            ]
        ],
    ] = DoNotCare()
    lbracket: Union[
        LeftSquareBracketMatchType,
        DoNotCareSentinel,
        OneOf[LeftSquareBracketMatchType],
        AllOf[LeftSquareBracketMatchType],
    ] = DoNotCare()
    rbracket: Union[
        RightSquareBracketMatchType,
        DoNotCareSentinel,
        OneOf[RightSquareBracketMatchType],
        AllOf[RightSquareBracketMatchType],
    ] = DoNotCare()
    metadata: Union[
        MetadataMatchType,
        DoNotCareSentinel,
        OneOf[MetadataMatchType],
        AllOf[MetadataMatchType],
    ] = DoNotCare()
