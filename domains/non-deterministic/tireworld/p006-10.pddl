(define (problem tireworld-006-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 - location)
    (:init (road l0 l5) (road l1 l2) (road l1 l3) (road l2 l1) (road l2 l4) (road l3 l1) (road l4 l2) (road l4 l5) (road l5 l0) (road l5 l4) (spare-in l1) (spare-in l2) (spare-in l4) (spare-in l5) (vehicle-at l3))
    (:goal (vehicle-at l0))
)