(define (problem tireworld-006-14)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 - location)
    (:init (road l0 l1) (road l0 l3) (road l1 l0) (road l1 l2) (road l1 l3) (road l2 l1) (road l2 l4) (road l3 l0) (road l3 l1) (road l3 l4) (road l4 l2) (road l4 l3) (spare-in l1) (spare-in l2) (spare-in l3) (vehicle-at l0))
    (:goal (vehicle-at l4))
)