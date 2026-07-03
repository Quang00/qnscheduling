graph [
  name "3_equal_paths"
  directed 0
  stats [
    nodes 8
    links 9
    min_degree 2
    max_degree 3
  ]

  node [ id 0 label "A" lon 0.0 lat 3.0 ]
  node [ id 1 label "B" lon 0.0 lat -3.0 ]

  node [ id 2 label "R1" lon -2.0 lat 1.0 ]
  node [ id 3 label "R2" lon -2.0 lat -1.0 ]

  node [ id 4 label "R3" lon 0.0 lat 1.0 ]
  node [ id 5 label "R4" lon 0.0 lat -1.0 ]

  node [ id 6 label "R5" lon 2.0 lat 1.0 ]
  node [ id 7 label "R6" lon 2.0 lat -1.0 ]

  edge [ source 0 target 2 dist 1.0 ]
  edge [ source 2 target 3 dist 1.0 ]
  edge [ source 3 target 1 dist 1.0 ]

  edge [ source 0 target 4 dist 5.0 ]
  edge [ source 4 target 5 dist 5.0 ]
  edge [ source 5 target 1 dist 5.0 ]

  edge [ source 0 target 6 dist 10.0 ]
  edge [ source 6 target 7 dist 10.0 ]
  edge [ source 7 target 1 dist 10.0 ]
]
