(define (problem tireworld-011-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l2) (road l1 l5) (road l1 l6) (road l10 l3) (road l10 l8) (road l2 l0) (road l2 l8) (road l3 l10) (road l3 l4) (road l4 l3) (road l4 l7) (road l5 l1) (road l5 l8) (road l5 l9) (road l6 l1) (road l6 l7) (road l7 l4) (road l7 l6) (road l7 l9) (road l8 l10) (road l8 l2) (road l8 l5) (road l9 l5) (road l9 l7) (spare-in l1) (spare-in l2) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l0))
    (:goal (vehicle-at l3))
)