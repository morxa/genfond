(define (problem p_s-3_n-3_l-3)
 (:domain spanner)
 (:objects 
     bob - man
     spanner1 spanner2 spanner3 - spanner
     nut1 nut2 nut3 - nut
     location1 location2 location3 - location
     shed gate - location
    )
 (:init 
    (at bob shed)
    (at spanner1 location1)
    (useable spanner1)
    (at spanner2 location2)
    (useable spanner2)
    (at spanner3 location1)
    (useable spanner3)
    (loose nut1)
    (at nut1 gate)
    (loose nut2)
    (at nut2 gate)
    (loose nut3)
    (at nut3 gate)
    (link shed location1)
    (link location3 gate)
    (link location1 location2)
    (link location2 location3)
)
 (:goal
  (and
   (tightened nut1)
   (tightened nut2)
   (tightened nut3)
)))
