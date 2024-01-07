(define (problem tireworld-006-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 - location)
    (:init (road l0 l4) (road l0 l5) (road l1 l4) (road l2 l5) (road l3 l2) (road l4 l0) (road l4 l1) (road l4 l3) (spare-in l0) (spare-in l4) (vehicle-at l1))
    (:goal (vehicle-at l5))
)