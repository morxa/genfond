(define (problem tireworld-012-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l1 l0) (road l1 l2) (road l10 l9) (road l2 l1) (road l2 l3) (road l3 l2) (road l3 l4) (road l4 l3) (road l4 l5) (road l5 l4) (road l5 l7) (road l6 l7) (road l7 l5) (road l7 l6) (road l7 l8) (road l8 l7) (road l8 l9) (road l9 l10) (road l9 l8) (spare-in l7) (vehicle-at l5))
    (:goal (vehicle-at l6))
)