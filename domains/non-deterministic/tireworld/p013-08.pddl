(define (problem tireworld-013-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l1 l0) (road l1 l10) (road l1 l11) (road l1 l2) (road l10 l1) (road l10 l11) (road l10 l12) (road l10 l9) (road l11 l1) (road l11 l10) (road l11 l12) (road l11 l3) (road l12 l10) (road l12 l11) (road l12 l5) (road l2 l1) (road l2 l3) (road l2 l4) (road l2 l9) (road l3 l11) (road l3 l2) (road l3 l9) (road l4 l2) (road l4 l7) (road l5 l12) (road l5 l6) (road l6 l5) (road l7 l4) (road l8 l9) (road l9 l10) (road l9 l2) (road l9 l3) (road l9 l8) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l9) (vehicle-at l5))
    (:goal (vehicle-at l7))
)