(define (problem tireworld-006-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 - location)
    (:init (road l0 l2) (road l0 l4) (road l1 l2) (road l1 l3) (road l1 l5) (road l2 l1) (road l3 l0) (road l4 l0) (road l4 l2) (road l4 l5) (road l5 l1) (road l5 l4) (spare-in l4) (spare-in l5) (vehicle-at l0))
    (:goal (vehicle-at l2))
)