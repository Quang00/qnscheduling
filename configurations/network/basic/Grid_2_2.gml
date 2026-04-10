graph [
    name "grid_2x2"
    directed 0
    stats [
        nodes 9
        links 12
        min_degree 2
        max_degree 4
    ]

    node [ id 0 label "A" lon -1.0 lat  1.0 ]
    node [ id 1 label "B" lon  0.0 lat  1.0 ]
    node [ id 2 label "C" lon  1.0 lat  1.0 ]

    node [ id 3 label "D" lon -1.0 lat  0.0 ]
    node [ id 4 label "E" lon  0.0 lat  0.0 ]
    node [ id 5 label "F" lon  1.0 lat  0.0 ]

    node [ id 6 label "G" lon -1.0 lat -1.0 ]
    node [ id 7 label "H" lon  0.0 lat -1.0 ]
    node [ id 8 label "I" lon  1.0 lat -1.0 ]

    edge [ source 0 target 1 dist 5.0 ]
    edge [ source 1 target 2 dist 5.0 ]

    edge [ source 3 target 4 dist 5.0 ]
    edge [ source 4 target 5 dist 5.0 ]

    edge [ source 6 target 7 dist 5.0 ]
    edge [ source 7 target 8 dist 5.0 ]

    edge [ source 0 target 3 dist 5.0 ]
    edge [ source 3 target 6 dist 5.0 ]

    edge [ source 1 target 4 dist 5.0 ]
    edge [ source 4 target 7 dist 5.0 ]

    edge [ source 2 target 5 dist 5.0 ]
    edge [ source 5 target 8 dist 5.0 ]
]