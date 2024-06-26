(define (problem tireworld-013-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l0 l7) (road l1 l6) (road l1 l9) (road l10 l3) (road l10 l5) (road l11 l5) (road l11 l9) (road l12 l0) (road l12 l5) (road l2 l6) (road l3 l10) (road l3 l4) (road l4 l3) (road l4 l7) (road l5 l10) (road l5 l11) (road l5 l12) (road l6 l1) (road l6 l2) (road l6 l8) (road l7 l0) (road l7 l4) (road l7 l8) (road l8 l6) (road l8 l7) (road l9 l1) (road l9 l11) (spare-in l0) (spare-in l1) (spare-in l12) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l8) (spare-in l9) (vehicle-at l7))
    (:goal (vehicle-at l2))
)