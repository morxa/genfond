(define (problem tireworld-015-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l3) (road l0 l4) (road l1 l10) (road l1 l3) (road l10 l1) (road l10 l12) (road l11 l6) (road l11 l9) (road l12 l10) (road l12 l6) (road l13 l14) (road l14 l13) (road l14 l6) (road l2 l6) (road l2 l8) (road l3 l0) (road l3 l1) (road l4 l0) (road l4 l5) (road l5 l4) (road l5 l7) (road l5 l9) (road l6 l11) (road l6 l12) (road l6 l14) (road l6 l2) (road l7 l5) (road l7 l8) (road l8 l2) (road l8 l7) (road l9 l11) (road l9 l5) (spare-in l14) (spare-in l2) (spare-in l6) (spare-in l8) (vehicle-at l13))
    (:goal (vehicle-at l7))
)