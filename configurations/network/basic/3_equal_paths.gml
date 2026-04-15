graph [
  name "3_equal_paths"
  directed 0

  node [ id 0 label "A" lon 0.0 lat 1.0 ]
  node [ id 2 label "E" lon 0.0 lat -1.0 ]

  node [ id 1 label "B" lon -1.0 lat 0.0 ]
  node [ id 3 label "C" lon 0.0 lat 0.0 ]
  node [ id 4 label "D" lon 1.0 lat 0.0 ]

  edge [ source 0 target 1 dist 1.0 ]
  edge [ source 1 target 2 dist 1.0 ]

  edge [ source 0 target 3 dist 1.0 ]
  edge [ source 3 target 2 dist 1.0 ]

  edge [ source 0 target 4 dist 1.0 ]
  edge [ source 4 target 2 dist 1.0 ]
]
