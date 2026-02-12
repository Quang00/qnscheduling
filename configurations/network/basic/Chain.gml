graph [
  name "chain_5"
  directed 0
  stats [
    nodes 5
    links 4
    min_degree 1
    max_degree 2
  ]

  node [ id 0 label "A" lon -1.0 lat 0.0 ]
  node [ id 1 label "C" lon 0.0  lat 0.0 ]
  node [ id 2 label "B" lon 3.0  lat 0.0 ]
  node [ id 3 label "D" lon 1.0  lat 0.0 ]
  node [ id 4 label "E" lon 2.0  lat 0.0 ]

  edge [ source 0 target 1 dist 5.0 ]
  edge [ source 1 target 3 dist 5.0 ]
  edge [ source 3 target 4 dist 5.0 ]
  edge [ source 4 target 2 dist 5.0 ]
]
