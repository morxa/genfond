(define (problem tireworld-013-1)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l5) (road l0 l9) (road l1 l0) (road l1 l2) (road l10 l11) (road l10 l9) (road l11 l10) (road l11 l12) (road l12 l11) (road l2 l1) (road l2 l3) (road l3 l2) (road l4 l5) (road l5 l0) (road l5 l4) (road l5 l6) (road l5 l8) (road l6 l5) (road l6 l7) (road l7 l6) (road l7 l8) (road l8 l5) (road l8 l7) (road l8 l9) (road l9 l0) (road l9 l10) (road l9 l8) (spare-in l0) (spare-in l5) (spare-in l8) (vehicle-at l9))
    (:goal (vehicle-at l7))
)