# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This file was generated by libcst.codegen.gen_matcher_classes
from dataclasses import dataclass
from typing import Sequence, Union

import libcst as cst
from libcst.matchers._match_types import (
    boolMatchType,
    EmptyLineMatchType,
    MetadataMatchType,
    SimpleStatementLineOrBaseCompoundStatementMatchType,
    strMatchType,
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
class Module(BaseMatcherNode):
    body: Union[
        Sequence[
            Union[
                SimpleStatementLineOrBaseCompoundStatementMatchType,
                DoNotCareSentinel,
                OneOf[SimpleStatementLineOrBaseCompoundStatementMatchType],
                AllOf[SimpleStatementLineOrBaseCompoundStatementMatchType],
                AtLeastN[
                    Union[
                        SimpleStatementLineOrBaseCompoundStatementMatchType,
                        DoNotCareSentinel,
                        OneOf[SimpleStatementLineOrBaseCompoundStatementMatchType],
                        AllOf[SimpleStatementLineOrBaseCompoundStatementMatchType],
                    ]
                ],
                AtMostN[
                    Union[
                        SimpleStatementLineOrBaseCompoundStatementMatchType,
                        DoNotCareSentinel,
                        OneOf[SimpleStatementLineOrBaseCompoundStatementMatchType],
                        AllOf[SimpleStatementLineOrBaseCompoundStatementMatchType],
                    ]
                ],
            ]
        ],
        DoNotCareSentinel,
        MatchIfTrue[
            Sequence[
                Union[
                    cst.SimpleStatementLine,
                    cst.BaseCompoundStatement,
                    OneOf[Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]],
                    AllOf[Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]],
                ]
            ]
        ],
        OneOf[
            Union[
                Sequence[
                    Union[
                        SimpleStatementLineOrBaseCompoundStatementMatchType,
                        OneOf[SimpleStatementLineOrBaseCompoundStatementMatchType],
                        AllOf[SimpleStatementLineOrBaseCompoundStatementMatchType],
                        AtLeastN[
                            Union[
                                SimpleStatementLineOrBaseCompoundStatementMatchType,
                                OneOf[
                                    SimpleStatementLineOrBaseCompoundStatementMatchType
                                ],
                                AllOf[
                                    SimpleStatementLineOrBaseCompoundStatementMatchType
                                ],
                            ]
                        ],
                        AtMostN[
                            Union[
                                SimpleStatementLineOrBaseCompoundStatementMatchType,
                                OneOf[
                                    SimpleStatementLineOrBaseCompoundStatementMatchType
                                ],
                                AllOf[
                                    SimpleStatementLineOrBaseCompoundStatementMatchType
                                ],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[
                    Sequence[
                        Union[
                            cst.SimpleStatementLine,
                            cst.BaseCompoundStatement,
                            OneOf[
                                Union[
                                    cst.SimpleStatementLine, cst.BaseCompoundStatement
                                ]
                            ],
                            AllOf[
                                Union[
                                    cst.SimpleStatementLine, cst.BaseCompoundStatement
                                ]
                            ],
                        ]
                    ]
                ],
            ]
        ],
        AllOf[
            Union[
                Sequence[
                    Union[
                        SimpleStatementLineOrBaseCompoundStatementMatchType,
                        OneOf[SimpleStatementLineOrBaseCompoundStatementMatchType],
                        AllOf[SimpleStatementLineOrBaseCompoundStatementMatchType],
                        AtLeastN[
                            Union[
                                SimpleStatementLineOrBaseCompoundStatementMatchType,
                                OneOf[
                                    SimpleStatementLineOrBaseCompoundStatementMatchType
                                ],
                                AllOf[
                                    SimpleStatementLineOrBaseCompoundStatementMatchType
                                ],
                            ]
                        ],
                        AtMostN[
                            Union[
                                SimpleStatementLineOrBaseCompoundStatementMatchType,
                                OneOf[
                                    SimpleStatementLineOrBaseCompoundStatementMatchType
                                ],
                                AllOf[
                                    SimpleStatementLineOrBaseCompoundStatementMatchType
                                ],
                            ]
                        ],
                    ]
                ],
                MatchIfTrue[
                    Sequence[
                        Union[
                            cst.SimpleStatementLine,
                            cst.BaseCompoundStatement,
                            OneOf[
                                Union[
                                    cst.SimpleStatementLine, cst.BaseCompoundStatement
                                ]
                            ],
                            AllOf[
                                Union[
                                    cst.SimpleStatementLine, cst.BaseCompoundStatement
                                ]
                            ],
                        ]
                    ]
                ],
            ]
        ],
    ] = DoNotCare()
    header: Union[
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
    encoding: Union[
        strMatchType, DoNotCareSentinel, OneOf[strMatchType], AllOf[strMatchType]
    ] = DoNotCare()
    default_indent: Union[
        strMatchType, DoNotCareSentinel, OneOf[strMatchType], AllOf[strMatchType]
    ] = DoNotCare()
    default_newline: Union[
        strMatchType, DoNotCareSentinel, OneOf[strMatchType], AllOf[strMatchType]
    ] = DoNotCare()
    has_trailing_newline: Union[
        boolMatchType, DoNotCareSentinel, OneOf[boolMatchType], AllOf[boolMatchType]
    ] = DoNotCare()
    metadata: Union[
        MetadataMatchType,
        DoNotCareSentinel,
        OneOf[MetadataMatchType],
        AllOf[MetadataMatchType],
    ] = DoNotCare()
