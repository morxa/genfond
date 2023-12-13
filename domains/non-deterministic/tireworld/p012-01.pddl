(define (problem tireworld-012-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l11) (road l0 l6) (road l1 l2) (road l10 l11) (road l10 l7) (road l10 l8) (road l10 l9) (road l11 l0) (road l11 l10) (road l2 l1) (road l2 l3) (road l2 l6) (road l3 l2) (road l3 l4) (road l4 l3) (road l4 l5) (road l5 l4) (road l5 l6) (road l5 l7) (road l6 l0) (road l6 l2) (road l6 l5) (road l6 l7) (road l7 l10) (road l7 l5) (road l7 l6) (road l7 l8) (road l8 l10) (road l8 l7) (road l8 l9) (road l9 l10) (road l9 l8) (spare-in l0) (spare-in l10) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (vehicle-at l8))
    (:goal (vehicle-at l11))
)