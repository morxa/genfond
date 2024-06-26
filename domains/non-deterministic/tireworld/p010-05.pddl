(define (problem tireworld-010-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l3) (road l0 l7) (road l1 l4) (road l1 l7) (road l1 l9) (road l2 l7) (road l2 l9) (road l3 l0) (road l3 l6) (road l4 l1) (road l4 l6) (road l5 l7) (road l5 l8) (road l6 l3) (road l6 l4) (road l7 l0) (road l7 l1) (road l7 l2) (road l7 l5) (road l8 l5) (road l8 l9) (road l9 l1) (road l9 l2) (road l9 l8) (spare-in l2) (spare-in l4) (spare-in l5) (spare-in l9) (vehicle-at l7))
    (:goal (vehicle-at l1))
)