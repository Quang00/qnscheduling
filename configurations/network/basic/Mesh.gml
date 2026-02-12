graph [
    name "complete_mesh_6"
    directed 0
    stats [
        nodes 6
        links 15
        min_degree 5
        max_degree 5
    ]

    node [ id 0 label "A" lon  2.0 lat  0.0 ]
    node [ id 1 label "B" lon  1.0 lat  1.732051 ]
    node [ id 2 label "C" lon -1.0 lat  1.732051 ]
    node [ id 3 label "D" lon -2.0 lat  0.0 ]
    node [ id 4 label "E" lon -1.0 lat -1.732051 ]
    node [ id 5 label "F" lon  1.0 lat -1.732051 ]

    edge [ source 0 target 1 dist 5.0 ]
    edge [ source 0 target 2 dist 5.0 ]
    edge [ source 0 target 3 dist 5.0 ]
    edge [ source 0 target 4 dist 5.0 ]
    edge [ source 0 target 5 dist 5.0 ]

    edge [ source 1 target 2 dist 5.0 ]
    edge [ source 1 target 3 dist 5.0 ]
    edge [ source 1 target 4 dist 5.0 ]
    edge [ source 1 target 5 dist 5.0 ]

    edge [ source 2 target 3 dist 5.0 ]
    edge [ source 2 target 4 dist 5.0 ]
    edge [ source 2 target 5 dist 5.0 ]

    edge [ source 3 target 4 dist 5.0 ]
    edge [ source 3 target 5 dist 5.0 ]

    edge [ source 4 target 5 dist 5.0 ]
]
