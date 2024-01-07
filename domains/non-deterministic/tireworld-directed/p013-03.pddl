(define (problem tireworld-013-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l0 l4) (road l1 l2) (road l10 l7) (road l10 l8) (road l11 l12) (road l11 l3) (road l12 l0) (road l12 l11) (road l12 l2) (road l12 l6) (road l2 l1) (road l2 l12) (road l3 l11) (road l3 l4) (road l4 l0) (road l4 l3) (road l4 l5) (road l5 l4) (road l5 l8) (road l6 l12) (road l6 l9) (road l7 l10) (road l7 l9) (road l8 l10) (road l8 l5) (road l9 l6) (road l9 l7) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l2) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l3))
    (:goal (vehicle-at l1))
)