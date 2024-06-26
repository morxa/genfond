(define (problem tireworld-011-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l2) (road l0 l8) (road l1 l3) (road l1 l9) (road l10 l2) (road l10 l7) (road l2 l0) (road l2 l10) (road l2 l6) (road l3 l1) (road l3 l4) (road l4 l3) (road l4 l5) (road l4 l7) (road l5 l4) (road l6 l2) (road l6 l8) (road l7 l10) (road l7 l4) (road l8 l0) (road l8 l6) (road l8 l9) (road l9 l1) (road l9 l8) (spare-in l1) (spare-in l10) (spare-in l3) (spare-in l4) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l2))
    (:goal (vehicle-at l5))
)