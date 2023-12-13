(define (problem tireworld-009-4)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 l6 l7 l8 - location)
    (:init (road l0 l1) (road l0 l4) (road l0 l7) (road l1 l0) (road l1 l2) (road l2 l1) (road l2 l3) (road l2 l5) (road l3 l2) (road l3 l4) (road l3 l8) (road l4 l0) (road l4 l3) (road l4 l5) (road l4 l6) (road l5 l2) (road l5 l4) (road l5 l6) (road l6 l4) (road l6 l5) (road l6 l7) (road l7 l0) (road l7 l6) (road l7 l8) (road l8 l3) (road l8 l7) (spare-in l0) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l2))
    (:goal (vehicle-at l3))
)