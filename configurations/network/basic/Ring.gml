graph [
    name "ring_7"
    directed 0
    stats [
        nodes 7
        links 7
        min_degree 2
        max_degree 2
    ]

    node [ id 0 label "A" lon  2.000000 lat  0.000000 ]
    node [ id 1 label "B" lon  1.246980 lat  1.563663 ]
    node [ id 2 label "C" lon -0.445042 lat  1.949856 ]
    node [ id 3 label "D" lon -1.801938 lat  0.867767 ]
    node [ id 4 label "E" lon -1.801938 lat -0.867767 ]
    node [ id 5 label "F" lon -0.445042 lat -1.949856 ]
    node [ id 6 label "G" lon  1.246980 lat -1.563663 ]

    edge [ source 0 target 1 dist 5.0 ]
    edge [ source 1 target 2 dist 5.0 ]
    edge [ source 2 target 3 dist 5.0 ]
    edge [ source 3 target 4 dist 5.0 ]
    edge [ source 4 target 5 dist 5.0 ]
    edge [ source 5 target 6 dist 5.0 ]
    edge [ source 6 target 0 dist 5.0 ]
]
