(define (problem tireworld-008-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 l6 l7 - location)
    (:init (road l0 l3) (road l0 l7) (road l1 l2) (road l1 l7) (road l2 l1) (road l2 l5) (road l3 l0) (road l3 l6) (road l4 l5) (road l4 l6) (road l5 l2) (road l5 l4) (road l6 l3) (road l6 l4) (road l7 l0) (road l7 l1) (spare-in l0) (spare-in l1) (spare-in l2) (spare-in l4) (spare-in l5) (spare-in l6) (vehicle-at l3))
    (:goal (vehicle-at l7))
)