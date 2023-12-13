(define (problem tireworld-017-17)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l4) (road l0 l6) (road l1 l0) (road l10 l11) (road l10 l9) (road l11 l10) (road l11 l12) (road l12 l11) (road l12 l13) (road l13 l12) (road l14 l15) (road l15 l14) (road l15 l16) (road l16 l15) (road l16 l8) (road l4 l0) (road l4 l5) (road l4 l8) (road l5 l4) (road l5 l6) (road l6 l0) (road l6 l5) (road l6 l7) (road l7 l6) (road l7 l8) (road l8 l16) (road l8 l4) (road l8 l7) (road l8 l9) (road l9 l10) (road l9 l8) (spare-in l0) (spare-in l4) (spare-in l8) (vehicle-at l16))
    (:goal (vehicle-at l6))
)