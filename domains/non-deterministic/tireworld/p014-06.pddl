(define (problem tireworld-014-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l5) (road l0 l6) (road l1 l9) (road l10 l12) (road l10 l8) (road l11 l3) (road l11 l9) (road l12 l10) (road l12 l2) (road l13 l4) (road l13 l8) (road l2 l12) (road l2 l5) (road l3 l11) (road l3 l7) (road l4 l13) (road l4 l5) (road l5 l0) (road l5 l2) (road l5 l4) (road l6 l0) (road l6 l7) (road l7 l3) (road l7 l6) (road l8 l10) (road l8 l13) (road l9 l1) (road l9 l11) (spare-in l0) (spare-in l11) (spare-in l12) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l9) (vehicle-at l10))
    (:goal (vehicle-at l1))
)