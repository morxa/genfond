(define (problem tireworld-009-15)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 l6 l7 l8 - location)
    (:init (road l0 l1) (road l0 l3) (road l0 l6) (road l1 l0) (road l1 l2) (road l2 l1) (road l2 l3) (road l3 l0) (road l3 l2) (road l3 l4) (road l3 l8) (road l4 l3) (road l4 l5) (road l5 l4) (road l5 l6) (road l6 l0) (road l6 l5) (road l6 l7) (road l7 l6) (road l8 l3) (spare-in l0) (spare-in l3) (vehicle-at l6))
    (:goal (vehicle-at l8))
)