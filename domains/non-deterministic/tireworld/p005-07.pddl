(define (problem tireworld-005-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 - location)
    (:init (road l0 l1) (road l0 l3) (road l0 l4) (road l1 l0) (road l1 l4) (road l2 l3) (road l2 l4) (road l3 l0) (road l3 l2) (road l4 l0) (road l4 l1) (road l4 l2) (spare-in l1) (spare-in l4) (vehicle-at l0))
    (:goal (vehicle-at l2))
)