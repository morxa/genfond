(define (problem tireworld-002-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 - location)
    (:init (road l0 l1) (road l1 l0) (vehicle-at l0))
    (:goal (vehicle-at l1))
)