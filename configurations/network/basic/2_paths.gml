graph [
  name "2_paths"
  directed 0
  stats [
    nodes 5
    links 5
    min_degree 2
    max_degree 2
  ]

  node [ id 0 label "B" lon 0.0 lat 0.0 ]
  node [ id 1 label "A" lon 0.5 lat 0.866 ]
  node [ id 2 label "D" lon 0.0 lat -1.0 ]
  node [ id 3 label "E" lon 1.0 lat -1.0 ]
  node [ id 4 label "C" lon 1.0 lat 0.0 ]

  edge [ source 0 target 1 dist 1.0 ]
  edge [ source 1 target 4 dist 1.0 ]

  edge [ source 0 target 2 dist 1.0 ]
  edge [ source 2 target 3 dist 1.0 ]
  edge [ source 3 target 4 dist 1.0 ]
]
