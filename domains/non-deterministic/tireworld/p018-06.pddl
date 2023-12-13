(define (problem tireworld-018-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l13) (road l0 l8) (road l1 l15) (road l1 l6) (road l10 l15) (road l10 l16) (road l11 l13) (road l11 l3) (road l12 l16) (road l12 l17) (road l13 l0) (road l13 l11) (road l13 l14) (road l13 l6) (road l14 l13) (road l14 l17) (road l14 l5) (road l15 l1) (road l15 l10) (road l15 l5) (road l15 l7) (road l16 l10) (road l16 l12) (road l17 l12) (road l17 l14) (road l17 l4) (road l17 l9) (road l2 l4) (road l2 l7) (road l3 l11) (road l3 l9) (road l4 l17) (road l4 l2) (road l5 l14) (road l5 l15) (road l6 l1) (road l6 l13) (road l7 l15) (road l7 l2) (road l8 l0) (road l9 l17) (road l9 l3) (spare-in l0) (spare-in l11) (spare-in l13) (spare-in l17) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l7) (spare-in l9) (vehicle-at l15))
    (:goal (vehicle-at l8))
)