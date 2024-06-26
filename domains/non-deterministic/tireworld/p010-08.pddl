(define (problem tireworld-010-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l3) (road l1 l5) (road l1 l9) (road l2 l6) (road l2 l8) (road l3 l0) (road l3 l4) (road l3 l7) (road l3 l9) (road l4 l3) (road l4 l6) (road l5 l1) (road l5 l7) (road l5 l8) (road l6 l2) (road l6 l4) (road l7 l3) (road l7 l5) (road l8 l2) (road l8 l5) (road l9 l1) (road l9 l3) (spare-in l1) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l7) (spare-in l9) (vehicle-at l5))
    (:goal (vehicle-at l0))
)