(define (problem tireworld-012-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l4) (road l1 l5) (road l10 l8) (road l11 l2) (road l11 l6) (road l2 l1) (road l3 l7) (road l3 l8) (road l4 l0) (road l5 l1) (road l5 l10) (road l5 l6) (road l6 l11) (road l6 l5) (road l7 l4) (road l8 l3) (road l9 l1) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l7) (spare-in l8) (vehicle-at l9))
    (:goal (vehicle-at l0))
)