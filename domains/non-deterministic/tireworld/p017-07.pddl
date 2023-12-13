(define (problem tireworld-017-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l11) (road l0 l7) (road l1 l5) (road l1 l9) (road l10 l12) (road l11 l0) (road l11 l5) (road l11 l6) (road l12 l10) (road l12 l16) (road l13 l3) (road l13 l6) (road l14 l4) (road l14 l8) (road l15 l3) (road l15 l5) (road l16 l12) (road l16 l9) (road l2 l6) (road l2 l7) (road l3 l13) (road l3 l15) (road l4 l14) (road l4 l7) (road l5 l1) (road l5 l11) (road l5 l15) (road l6 l11) (road l6 l13) (road l6 l2) (road l7 l0) (road l7 l2) (road l7 l4) (road l8 l14) (road l9 l1) (road l9 l16) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l16) (spare-in l2) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l9) (vehicle-at l8))
    (:goal (vehicle-at l10))
)