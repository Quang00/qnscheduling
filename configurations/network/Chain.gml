graph [
  name "chain"
  directed 0
  stats [
    nodes 3
    links 2
    min_degree 1
    max_degree 4
  ]

  node [ id 0 label "A" lon -1.0 lat 0.0 ]
  node [ id 1 label "R" lon 1.0 lat 0.0 ]
  node [ id 2 label "B" lon 3.0 lat 0.0 ]

  edge [ source 0 target 1 dist 5.0 ]
  edge [ source 1 target 2 dist 5.0 ]

]
