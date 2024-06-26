(define (problem tireworld-011-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l8) (road l0 l9) (road l1 l2) (road l1 l3) (road l10 l4) (road l10 l5) (road l2 l1) (road l2 l6) (road l3 l1) (road l3 l4) (road l3 l7) (road l3 l9) (road l4 l10) (road l4 l3) (road l5 l10) (road l5 l7) (road l6 l2) (road l6 l8) (road l7 l3) (road l7 l5) (road l8 l0) (road l8 l6) (road l9 l0) (road l9 l3) (spare-in l1) (spare-in l3) (spare-in l5) (spare-in l7) (vehicle-at l10))
    (:goal (vehicle-at l2))
)