(define (problem tireworld-006-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 - location)
    (:init (road l0 l1) (road l0 l3) (road l1 l0) (road l1 l4) (road l2 l3) (road l2 l5) (road l3 l0) (road l3 l2) (road l4 l1) (road l4 l5) (road l5 l2) (road l5 l4) (spare-in l0) (spare-in l1) (spare-in l2) (vehicle-at l3))
    (:goal (vehicle-at l4))
)