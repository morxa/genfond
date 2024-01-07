(define (problem tireworld-015-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l5) (road l1 l0) (road l1 l13) (road l10 l12) (road l10 l14) (road l11 l4) (road l11 l5) (road l12 l10) (road l12 l4) (road l13 l1) (road l14 l10) (road l14 l7) (road l2 l3) (road l2 l4) (road l2 l6) (road l2 l7) (road l3 l2) (road l3 l4) (road l4 l11) (road l4 l12) (road l4 l2) (road l4 l3) (road l5 l0) (road l5 l11) (road l6 l2) (road l6 l8) (road l7 l14) (road l7 l2) (road l8 l6) (road l8 l9) (road l9 l13) (road l9 l8) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l3) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l4))
    (:goal (vehicle-at l2))
)