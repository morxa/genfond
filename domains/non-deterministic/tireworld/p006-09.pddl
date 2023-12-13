(define (problem tireworld-006-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 - location)
    (:init (road l0 l3) (road l0 l5) (road l1 l4) (road l1 l5) (road l2 l3) (road l2 l4) (road l3 l0) (road l3 l2) (road l4 l1) (road l4 l2) (road l5 l0) (road l5 l1) (spare-in l0) (spare-in l1) (spare-in l3) (spare-in l4) (vehicle-at l2))
    (:goal (vehicle-at l5))
)