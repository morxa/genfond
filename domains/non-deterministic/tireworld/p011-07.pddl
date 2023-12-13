(define (problem tireworld-011-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l2) (road l0 l4) (road l1 l10) (road l10 l1) (road l10 l7) (road l10 l8) (road l2 l0) (road l2 l6) (road l2 l9) (road l3 l6) (road l4 l0) (road l4 l8) (road l5 l7) (road l5 l9) (road l6 l2) (road l6 l3) (road l7 l10) (road l7 l5) (road l8 l10) (road l8 l4) (road l9 l2) (road l9 l5) (spare-in l10) (spare-in l2) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l9) (vehicle-at l1))
    (:goal (vehicle-at l3))
)