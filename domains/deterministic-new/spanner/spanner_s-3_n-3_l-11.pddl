(define (problem p_s-3_n-3_l-11)
 (:domain spanner)
 (:objects 
     bob - man
     spanner1 spanner2 spanner3 - spanner
     nut1 nut2 nut3 - nut
     location1 location2 location3 location4 location5 location6 location7 location8 location9 location10 location11 - location
     shed gate - location
    )
 (:init 
    (at bob shed)
    (at spanner1 location3)
    (useable spanner1)
    (at spanner2 location1)
    (useable spanner2)
    (at spanner3 location2)
    (useable spanner3)
    (loose nut1)
    (at nut1 gate)
    (loose nut2)
    (at nut2 gate)
    (loose nut3)
    (at nut3 gate)
    (link shed location1)
    (link location11 gate)
    (link location1 location2)
    (link location2 location3)
    (link location3 location4)
    (link location4 location5)
    (link location5 location6)
    (link location6 location7)
    (link location7 location8)
    (link location8 location9)
    (link location9 location10)
    (link location10 location11)
)
 (:goal
  (and
   (tightened nut1)
   (tightened nut2)
   (tightened nut3)
)))
