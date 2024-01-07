(define (problem tireworld-007-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 l6 - location)
    (:init (road l0 l1) (road l0 l6) (road l1 l4) (road l1 l5) (road l2 l5) (road l3 l0) (road l4 l1) (road l4 l5) (road l6 l0) (road l6 l2) (spare-in l0) (spare-in l1) (spare-in l2) (spare-in l4) (spare-in l6) (vehicle-at l3))
    (:goal (vehicle-at l5))
)