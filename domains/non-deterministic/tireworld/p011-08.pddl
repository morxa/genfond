(define (problem tireworld-011-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l3) (road l0 l4) (road l1 l10) (road l1 l6) (road l10 l1) (road l10 l7) (road l10 l9) (road l2 l6) (road l2 l8) (road l3 l0) (road l3 l5) (road l4 l0) (road l4 l8) (road l5 l3) (road l6 l1) (road l6 l2) (road l7 l10) (road l7 l9) (road l8 l2) (road l8 l4) (road l9 l10) (road l9 l7) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l6) (spare-in l8) (vehicle-at l7))
    (:goal (vehicle-at l5))
)