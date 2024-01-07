(define (problem tireworld-016-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l1 l11) (road l10 l14) (road l10 l9) (road l11 l10) (road l12 l5) (road l13 l7) (road l14 l13) (road l15 l0) (road l15 l1) (road l2 l15) (road l2 l8) (road l4 l5) (road l4 l8) (road l5 l4) (road l5 l6) (road l6 l4) (road l7 l5) (road l8 l2) (road l8 l3) (road l9 l0) (spare-in l10) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l4))
    (:goal (vehicle-at l3))
)