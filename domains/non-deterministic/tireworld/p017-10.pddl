(define (problem tireworld-017-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l11) (road l1 l0) (road l1 l2) (road l10 l11) (road l10 l8) (road l11 l0) (road l11 l10) (road l11 l12) (road l11 l13) (road l12 l11) (road l12 l13) (road l13 l11) (road l13 l12) (road l13 l14) (road l13 l9) (road l14 l13) (road l14 l16) (road l15 l16) (road l15 l3) (road l15 l8) (road l16 l14) (road l16 l15) (road l16 l2) (road l2 l1) (road l2 l16) (road l2 l3) (road l3 l15) (road l3 l2) (road l3 l4) (road l3 l7) (road l4 l3) (road l4 l5) (road l5 l4) (road l6 l7) (road l7 l3) (road l7 l6) (road l7 l8) (road l7 l9) (road l8 l10) (road l8 l15) (road l8 l7) (road l8 l9) (road l9 l13) (road l9 l7) (road l9 l8) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l13) (spare-in l15) (spare-in l16) (spare-in l2) (spare-in l3) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l10))
    (:goal (vehicle-at l14))
)