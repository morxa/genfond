(define (problem tireworld-007-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 l6 - location)
    (:init (road l0 l1) (road l0 l4) (road l1 l0) (road l1 l5) (road l2 l5) (road l3 l4) (road l3 l6) (road l4 l0) (road l4 l3) (road l5 l1) (road l5 l2) (road l5 l6) (road l6 l3) (road l6 l5) (spare-in l0) (spare-in l3) (spare-in l4) (spare-in l5) (vehicle-at l2))
    (:goal (vehicle-at l6))
)