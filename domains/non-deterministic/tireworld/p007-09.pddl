(define (problem tireworld-007-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 l6 - location)
    (:init (road l0 l1) (road l0 l4) (road l1 l0) (road l1 l2) (road l2 l1) (road l2 l3) (road l3 l2) (road l3 l5) (road l3 l6) (road l4 l0) (road l4 l5) (road l4 l6) (road l5 l3) (road l5 l4) (road l6 l3) (road l6 l4) (spare-in l2) (spare-in l3) (spare-in l6) (vehicle-at l4))
    (:goal (vehicle-at l1))
)