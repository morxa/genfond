(define (problem tireworld-010-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l2) (road l0 l6) (road l0 l7) (road l1 l0) (road l1 l5) (road l2 l0) (road l2 l3) (road l3 l2) (road l3 l9) (road l4 l6) (road l4 l7) (road l5 l1) (road l6 l0) (road l6 l4) (road l6 l9) (road l7 l0) (road l7 l4) (road l7 l8) (road l8 l7) (road l8 l9) (road l9 l3) (road l9 l6) (road l9 l8) (spare-in l0) (spare-in l1) (vehicle-at l2))
    (:goal (vehicle-at l5))
)