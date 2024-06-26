(define (problem tireworld-012-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l10) (road l0 l11) (road l0 l2) (road l0 l3) (road l1 l2) (road l1 l4) (road l10 l0) (road l10 l7) (road l10 l8) (road l10 l9) (road l11 l0) (road l11 l6) (road l2 l0) (road l2 l1) (road l2 l3) (road l3 l0) (road l3 l2) (road l3 l5) (road l3 l7) (road l3 l8) (road l4 l1) (road l4 l6) (road l5 l3) (road l5 l9) (road l6 l11) (road l6 l4) (road l7 l10) (road l7 l3) (road l8 l10) (road l8 l3) (road l9 l10) (road l9 l5) (spare-in l1) (spare-in l10) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l9) (vehicle-at l3))
    (:goal (vehicle-at l2))
)