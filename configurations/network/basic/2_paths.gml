graph [
  name "2_paths"
  directed 0
  stats [
    nodes 5
    links 5
    min_degree 2
    max_degree 2
  ]

  node [ id 0 label "A" lon 0.0 lat 1.0 ]
  node [ id 1 label "B" lon 0.0 lat -1.0 ]

  node [ id 2 label "R1" lon -1.0 lat 0.0 ]

  node [ id 3 label "R2" lon 1.0 lat 0.5 ]
  node [ id 4 label "R3" lon 1.0 lat -0.5 ]

  edge [ source 0 target 2 dist 5.0 ]
  edge [ source 2 target 1 dist 5.0 ]

  edge [ source 0 target 3 dist 5.0 ]
  edge [ source 3 target 4 dist 5.0 ]
  edge [ source 4 target 1 dist 5.0 ]
]
