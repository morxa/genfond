(define (problem p_s-2_n-1_l-7)
 (:domain spanner)
 (:objects 
     bob - man
     spanner1 spanner2 - spanner
     nut1 - nut
     location1 location2 location3 location4 location5 location6 location7 - location
     shed gate - location
    )
 (:init 
    (at bob shed)
    (at spanner1 location6)
    (useable spanner1)
    (at spanner2 location3)
    (useable spanner2)
    (loose nut1)
    (at nut1 gate)
    (link shed location1)
    (link location7 gate)
    (link location1 location2)
    (link location2 location3)
    (link location3 location4)
    (link location4 location5)
    (link location5 location6)
    (link location6 location7)
)
 (:goal
  (and
   (tightened nut1)
)))
