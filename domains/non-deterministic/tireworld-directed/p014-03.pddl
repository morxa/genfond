(define (problem tireworld-014-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l5) (road l1 l12) (road l10 l11) (road l10 l7) (road l11 l6) (road l12 l10) (road l13 l1) (road l2 l4) (road l3 l8) (road l4 l3) (road l6 l5) (road l7 l0) (road l8 l13) (road l9 l2) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l9))
    (:goal (vehicle-at l5))
)