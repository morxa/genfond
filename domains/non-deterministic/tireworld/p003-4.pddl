(define (problem tireworld-003-4)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 - location)
    (:init (road l0 l1) (road l1 l0) (road l1 l2) (road l2 l1) (vehicle-at l1))
    (:goal (vehicle-at l0))
)