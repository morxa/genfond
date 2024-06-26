(define (problem tireworld-015-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l11) (road l0 l6) (road l0 l8) (road l1 l13) (road l1 l4) (road l10 l12) (road l10 l3) (road l10 l6) (road l11 l0) (road l12 l10) (road l12 l7) (road l13 l1) (road l13 l7) (road l13 l8) (road l14 l7) (road l14 l8) (road l2 l5) (road l2 l7) (road l3 l10) (road l3 l7) (road l4 l1) (road l4 l9) (road l5 l2) (road l6 l0) (road l6 l10) (road l7 l12) (road l7 l13) (road l7 l14) (road l7 l2) (road l7 l3) (road l7 l9) (road l8 l0) (road l8 l13) (road l8 l14) (road l9 l4) (road l9 l7) (spare-in l0) (spare-in l10) (spare-in l14) (spare-in l2) (spare-in l3) (spare-in l6) (spare-in l7) (spare-in l9) (vehicle-at l5))
    (:goal (vehicle-at l11))
)