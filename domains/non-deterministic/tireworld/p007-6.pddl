(define (problem tireworld-007-6)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 l6 - location)
    (:init (road l0 l1) (road l0 l4) (road l0 l6) (road l1 l0) (road l1 l2) (road l1 l3) (road l1 l6) (road l2 l1) (road l2 l3) (road l2 l4) (road l3 l1) (road l3 l2) (road l3 l4) (road l4 l0) (road l4 l2) (road l4 l3) (road l4 l5) (road l5 l4) (road l5 l6) (road l6 l0) (road l6 l1) (road l6 l5) (spare-in l0) (spare-in l1) (spare-in l4) (spare-in l6) (vehicle-at l2))
    (:goal (vehicle-at l3))
)