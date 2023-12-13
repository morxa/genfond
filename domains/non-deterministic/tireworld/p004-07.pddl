(define (problem tireworld-004-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 - location)
    (:init (road l0 l1) (road l0 l2) (road l0 l3) (road l1 l0) (road l1 l2) (road l1 l3) (road l2 l0) (road l2 l1) (road l3 l0) (road l3 l1) (vehicle-at l0))
    (:goal (vehicle-at l3))
)