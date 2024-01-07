(define (problem tireworld-016-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l0 l6) (road l1 l4) (road l1 l6) (road l10 l4) (road l11 l12) (road l11 l9) (road l12 l0) (road l12 l11) (road l12 l7) (road l13 l3) (road l14 l15) (road l14 l7) (road l15 l2) (road l2 l13) (road l2 l15) (road l3 l5) (road l4 l10) (road l5 l6) (road l6 l0) (road l6 l1) (road l6 l12) (road l6 l5) (road l7 l12) (road l7 l14) (road l7 l8) (road l8 l7) (road l9 l4) (spare-in l11) (spare-in l12) (spare-in l4) (spare-in l7) (spare-in l9) (vehicle-at l8))
    (:goal (vehicle-at l10))
)