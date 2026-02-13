graph [
  name "chain_2"
  directed 0
  stats [
    nodes 3
    links 2
    min_degree 1
    max_degree 2
  ]

  node [ id 0 label "A" lon -1.0 lat 0.0 ]
  node [ id 1 label "R" lon  0.0 lat 0.0 ]
  node [ id 3 label "B" lon  1.0 lat 0.0 ]

  edge [ source 0 target 1 dist 5.0 ]
  edge [ source 1 target 3 dist 5.0 ]
]
