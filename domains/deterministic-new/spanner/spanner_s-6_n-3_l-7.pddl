(define (problem prob)
 (:domain spanner)
 (:objects 
     bob - man
     spanner1 spanner2 spanner3 spanner4 spanner5 spanner6 - spanner
     nut1 nut2 nut3 - nut
     location1 location2 location3 location4 location5 location6 location7 - location
     shed gate - location
    )
 (:init 
    (at bob shed)
    (at spanner1 location7)
    (useable spanner1)
    (at spanner2 location6)
    (useable spanner2)
    (at spanner3 location3)
    (useable spanner3)
    (at spanner4 location1)
    (useable spanner4)
    (at spanner5 location4)
    (useable spanner5)
    (at spanner6 location7)
    (useable spanner6)
    (loose nut1)
    (at nut1 gate)
    (loose nut2)
    (at nut2 gate)
    (loose nut3)
    (at nut3 gate)
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
   (tightened nut2)
   (tightened nut3)
)))
