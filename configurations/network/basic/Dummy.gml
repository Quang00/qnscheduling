graph [
  name "dummy"
  directed 0
  stats [
    nodes 6
    links 5
    min_degree 1
    max_degree 4
  ]

  node [ id 0 label "L" lon -1.0 lat 0.0 ]
  node [ id 1 label "R" lon 1.0 lat 0.0 ]

  node [ id 2 label "L1" lon -2.0 lat 1.0 ]
  node [ id 3 label "L2" lon -2.0 lat 0.0 ]

  node [ id 5 label "R1" lon 2.0 lat 1.0 ]
  node [ id 6 label "R2" lon 2.0 lat 0.0 ]

  edge [ source 2 target 0 dist 1.0 ]
  edge [ source 3 target 0 dist 2.0 ]

  edge [ source 0 target 1 dist 10.0 ]

  edge [ source 1 target 5 dist 1.0 ]
  edge [ source 1 target 6 dist 2.0 ]
]
