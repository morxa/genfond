(define (problem tireworld-015-7)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l11) (road l0 l14) (road l1 l2) (road l10 l11) (road l10 l9) (road l11 l0) (road l11 l10) (road l11 l12) (road l11 l8) (road l12 l11) (road l12 l13) (road l13 l12) (road l13 l14) (road l14 l0) (road l14 l13) (road l2 l1) (road l2 l7) (road l2 l8) (road l4 l5) (road l5 l4) (road l6 l7) (road l7 l2) (road l7 l6) (road l7 l8) (road l8 l11) (road l8 l2) (road l8 l7) (road l8 l9) (road l9 l10) (road l9 l8) (spare-in l0) (spare-in l11) (spare-in l14) (spare-in l2) (spare-in l8) (vehicle-at l7))
    (:goal (vehicle-at l13))
)