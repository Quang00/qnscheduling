graph [
  name "star_7"
  directed 0
  stats [
    nodes 7
    links 6
    min_degree 1
    max_degree 6
  ]

  node [ id 0 label "A" lon 0.0  lat 0.0 ]
  node [ id 1 label "B" lon 2.0  lat 0.0 ]
  node [ id 2 label "C" lon 1.0  lat 1.732 ]
  node [ id 3 label "D" lon -1.0 lat 1.732 ]
  node [ id 4 label "E" lon -2.0 lat 0.0 ]
  node [ id 5 label "F" lon -1.0 lat -1.732 ]
  node [ id 6 label "G" lon 1.0  lat -1.732 ]

  edge [ source 0 target 1 dist 5.0 ]
  edge [ source 0 target 2 dist 5.0 ]
  edge [ source 0 target 3 dist 5.0 ]
  edge [ source 0 target 4 dist 5.0 ]
  edge [ source 0 target 5 dist 5.0 ]
  edge [ source 0 target 6 dist 5.0 ]
]
