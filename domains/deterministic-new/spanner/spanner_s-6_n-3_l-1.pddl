(define (problem p_s-6_n-3_l-1)
 (:domain spanner)
 (:objects 
     bob - man
     spanner1 spanner2 spanner3 spanner4 spanner5 spanner6 - spanner
     nut1 nut2 nut3 - nut
     location1 - location
     shed gate - location
    )
 (:init 
    (at bob shed)
    (at spanner1 location1)
    (useable spanner1)
    (at spanner2 location1)
    (useable spanner2)
    (at spanner3 location1)
    (useable spanner3)
    (at spanner4 location1)
    (useable spanner4)
    (at spanner5 location1)
    (useable spanner5)
    (at spanner6 location1)
    (useable spanner6)
    (loose nut1)
    (at nut1 gate)
    (loose nut2)
    (at nut2 gate)
    (loose nut3)
    (at nut3 gate)
    (link shed location1)
    (link location1 gate)
)
 (:goal
  (and
   (tightened nut1)
   (tightened nut2)
   (tightened nut3)
)))
