(define (problem p_s-6_n-3_l-18)
 (:domain spanner)
 (:objects 
     bob - man
     spanner1 spanner2 spanner3 spanner4 spanner5 spanner6 - spanner
     nut1 nut2 nut3 - nut
     location1 location2 location3 location4 location5 location6 location7 location8 location9 location10 location11 location12 location13 location14 location15 location16 location17 location18 - location
     shed gate - location
    )
 (:init 
    (at bob shed)
    (at spanner1 location5)
    (useable spanner1)
    (at spanner2 location10)
    (useable spanner2)
    (at spanner3 location12)
    (useable spanner3)
    (at spanner4 location8)
    (useable spanner4)
    (at spanner5 location1)
    (useable spanner5)
    (at spanner6 location12)
    (useable spanner6)
    (loose nut1)
    (at nut1 gate)
    (loose nut2)
    (at nut2 gate)
    (loose nut3)
    (at nut3 gate)
    (link shed location1)
    (link location18 gate)
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
    (link location11 location12)
    (link location12 location13)
    (link location13 location14)
    (link location14 location15)
    (link location15 location16)
    (link location16 location17)
    (link location17 location18)
)
 (:goal
  (and
   (tightened nut1)
   (tightened nut2)
   (tightened nut3)
)))
