(define (problem tireworld-016-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l7) (road l0 l8) (road l1 l12) (road l1 l13) (road l1 l4) (road l10 l2) (road l10 l8) (road l11 l12) (road l11 l5) (road l12 l1) (road l12 l11) (road l13 l1) (road l13 l14) (road l13 l15) (road l13 l3) (road l13 l7) (road l13 l9) (road l14 l13) (road l15 l13) (road l15 l8) (road l2 l10) (road l2 l6) (road l3 l13) (road l3 l4) (road l4 l1) (road l4 l3) (road l4 l9) (road l5 l11) (road l5 l6) (road l6 l2) (road l6 l5) (road l7 l0) (road l7 l13) (road l8 l0) (road l8 l10) (road l8 l15) (road l9 l13) (road l9 l4) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l15) (spare-in l2) (spare-in l3) (spare-in l5) (spare-in l6) (spare-in l8) (spare-in l9) (vehicle-at l14))
    (:goal (vehicle-at l12))
)