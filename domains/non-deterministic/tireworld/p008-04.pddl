(define (problem tireworld-008-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 l6 l7 - location)
    (:init (road l0 l1) (road l0 l3) (road l1 l0) (road l1 l2) (road l1 l7) (road l2 l1) (road l3 l0) (road l3 l6) (road l4 l5) (road l4 l7) (road l5 l4) (road l5 l7) (road l6 l3) (road l6 l7) (road l7 l1) (road l7 l4) (road l7 l5) (road l7 l6) (spare-in l1) (spare-in l3) (spare-in l4) (spare-in l6) (spare-in l7) (vehicle-at l5))
    (:goal (vehicle-at l2))
)