(define (problem p_s-4_n-4_l-3)
 (:domain spanner)
 (:objects 
     bob - man
     spanner1 spanner2 spanner3 spanner4 - spanner
     nut1 nut2 nut3 nut4 - nut
     location1 location2 location3 - location
     shed gate - location
    )
 (:init 
    (at bob shed)
    (at spanner1 location2)
    (useable spanner1)
    (at spanner2 location3)
    (useable spanner2)
    (at spanner3 location1)
    (useable spanner3)
    (at spanner4 location2)
    (useable spanner4)
    (loose nut1)
    (at nut1 gate)
    (loose nut2)
    (at nut2 gate)
    (loose nut3)
    (at nut3 gate)
    (loose nut4)
    (at nut4 gate)
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
   (tightened nut4)
)))
