graph [
    name "grid_5x5"
    directed 0
    stats [
        nodes 25
        links 40
        min_degree 2
        max_degree 4
    ]

    node [ id 0  label "A" lon -2.0 lat  2.0 ]
    node [ id 1  label "B" lon -1.0 lat  2.0 ]
    node [ id 2  label "C" lon  0.0 lat  2.0 ]
    node [ id 3  label "D" lon  1.0 lat  2.0 ]
    node [ id 4  label "E" lon  2.0 lat  2.0 ]

    node [ id 5  label "F" lon -2.0 lat  1.0 ]
    node [ id 6  label "G" lon -1.0 lat  1.0 ]
    node [ id 7  label "H" lon  0.0 lat  1.0 ]
    node [ id 8  label "I" lon  1.0 lat  1.0 ]
    node [ id 9  label "J" lon  2.0 lat  1.0 ]

    node [ id 10 label "K" lon -2.0 lat  0.0 ]
    node [ id 11 label "L" lon -1.0 lat  0.0 ]
    node [ id 12 label "M" lon  0.0 lat  0.0 ]
    node [ id 13 label "N" lon  1.0 lat  0.0 ]
    node [ id 14 label "O" lon  2.0 lat  0.0 ]

    node [ id 15 label "P" lon -2.0 lat -1.0 ]
    node [ id 16 label "Q" lon -1.0 lat -1.0 ]
    node [ id 17 label "R" lon  0.0 lat -1.0 ]
    node [ id 18 label "S" lon  1.0 lat -1.0 ]
    node [ id 19 label "T" lon  2.0 lat -1.0 ]

    node [ id 20 label "U" lon -2.0 lat -2.0 ]
    node [ id 21 label "V" lon -1.0 lat -2.0 ]
    node [ id 22 label "W" lon  0.0 lat -2.0 ]
    node [ id 23 label "X" lon  1.0 lat -2.0 ]
    node [ id 24 label "Y" lon  2.0 lat -2.0 ]

    edge [ source 0 target 1 dist 5.0 ]
    edge [ source 1 target 2 dist 5.0 ]
    edge [ source 2 target 3 dist 5.0 ]
    edge [ source 3 target 4 dist 5.0 ]

    edge [ source 5 target 6 dist 5.0 ]
    edge [ source 6 target 7 dist 5.0 ]
    edge [ source 7 target 8 dist 5.0 ]
    edge [ source 8 target 9 dist 5.0 ]

    edge [ source 10 target 11 dist 5.0 ]
    edge [ source 11 target 12 dist 5.0 ]
    edge [ source 12 target 13 dist 5.0 ]
    edge [ source 13 target 14 dist 5.0 ]

    edge [ source 15 target 16 dist 5.0 ]
    edge [ source 16 target 17 dist 5.0 ]
    edge [ source 17 target 18 dist 5.0 ]
    edge [ source 18 target 19 dist 5.0 ]

    edge [ source 20 target 21 dist 5.0 ]
    edge [ source 21 target 22 dist 5.0 ]
    edge [ source 22 target 23 dist 5.0 ]
    edge [ source 23 target 24 dist 5.0 ]

    edge [ source 0 target 5 dist 5.0 ]
    edge [ source 5 target 10 dist 5.0 ]
    edge [ source 10 target 15 dist 5.0 ]
    edge [ source 15 target 20 dist 5.0 ]

    edge [ source 1 target 6 dist 5.0 ]
    edge [ source 6 target 11 dist 5.0 ]
    edge [ source 11 target 16 dist 5.0 ]
    edge [ source 16 target 21 dist 5.0 ]

    edge [ source 2 target 7 dist 5.0 ]
    edge [ source 7 target 12 dist 5.0 ]
    edge [ source 12 target 17 dist 5.0 ]
    edge [ source 17 target 22 dist 5.0 ]

    edge [ source 3 target 8 dist 5.0 ]
    edge [ source 8 target 13 dist 5.0 ]
    edge [ source 13 target 18 dist 5.0 ]
    edge [ source 18 target 23 dist 5.0 ]

    edge [ source 4 target 9 dist 5.0 ]
    edge [ source 9 target 14 dist 5.0 ]
    edge [ source 14 target 19 dist 5.0 ]
    edge [ source 19 target 24 dist 5.0 ]
]
