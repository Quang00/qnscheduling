graph [
    name "grid_3x3"
    directed 0
    stats [
        nodes 16
        links 24
        min_degree 2
        max_degree 4
    ]

    node [ id 0  label "A" lon -1.5 lat  1.5 ]
    node [ id 1  label "B" lon -0.5 lat  1.5 ]
    node [ id 2  label "C" lon  0.5 lat  1.5 ]
    node [ id 3  label "D" lon  1.5 lat  1.5 ]

    node [ id 4  label "E" lon -1.5 lat  0.5 ]
    node [ id 5  label "F" lon -0.5 lat  0.5 ]
    node [ id 6  label "G" lon  0.5 lat  0.5 ]
    node [ id 7  label "H" lon  1.5 lat  0.5 ]

    node [ id 8  label "I" lon -1.5 lat -0.5 ]
    node [ id 9  label "J" lon -0.5 lat -0.5 ]
    node [ id 10 label "K" lon  0.5 lat -0.5 ]
    node [ id 11 label "L" lon  1.5 lat -0.5 ]

    node [ id 12 label "M" lon -1.5 lat -1.5 ]
    node [ id 13 label "N" lon -0.5 lat -1.5 ]
    node [ id 14 label "O" lon  0.5 lat -1.5 ]
    node [ id 15 label "P" lon  1.5 lat -1.5 ]

    edge [ source 0 target 1 dist 1.0 ]
    edge [ source 1 target 2 dist 1.0 ]
    edge [ source 2 target 3 dist 1.0 ]

    edge [ source 4 target 5 dist 1.0 ]
    edge [ source 5 target 6 dist 1.0 ]
    edge [ source 6 target 7 dist 1.0 ]

    edge [ source 8 target 9 dist 1.0 ]
    edge [ source 9 target 10 dist 1.0 ]
    edge [ source 10 target 11 dist 1.0 ]

    edge [ source 12 target 13 dist 1.0 ]
    edge [ source 13 target 14 dist 1.0 ]
    edge [ source 14 target 15 dist 1.0 ]

    edge [ source 0 target 4 dist 1.0 ]
    edge [ source 4 target 8 dist 1.0 ]
    edge [ source 8 target 12 dist 1.0 ]

    edge [ source 1 target 5 dist 1.0 ]
    edge [ source 5 target 9 dist 1.0 ]
    edge [ source 9 target 13 dist 1.0 ]

    edge [ source 2 target 6 dist 1.0 ]
    edge [ source 6 target 10 dist 1.0 ]
    edge [ source 10 target 14 dist 1.0 ]

    edge [ source 3 target 7 dist 1.0 ]
    edge [ source 7 target 11 dist 1.0 ]
    edge [ source 11 target 15 dist 1.0 ]
]
