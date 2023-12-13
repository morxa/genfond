(define (problem tireworld-012-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l7) (road l1 l0) (road l1 l11) (road l1 l2) (road l10 l5) (road l10 l8) (road l11 l1) (road l11 l4) (road l11 l8) (road l2 l1) (road l2 l9) (road l3 l5) (road l4 l11) (road l4 l6) (road l5 l10) (road l5 l3) (road l5 l6) (road l6 l4) (road l6 l5) (road l7 l0) (road l7 l9) (road l8 l10) (road l8 l11) (road l9 l2) (road l9 l7) (spare-in l1) (spare-in l11) (spare-in l2) (spare-in l4) (spare-in l5) (spare-in l6) (vehicle-at l3))
    (:goal (vehicle-at l9))
)