(define (problem tireworld-018-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l10) (road l0 l7) (road l1 l14) (road l1 l17) (road l10 l0) (road l10 l12) (road l11 l13) (road l11 l16) (road l12 l10) (road l12 l14) (road l13 l11) (road l13 l6) (road l14 l1) (road l14 l12) (road l14 l16) (road l14 l3) (road l15 l17) (road l15 l8) (road l16 l11) (road l16 l14) (road l16 l4) (road l17 l1) (road l17 l15) (road l17 l5) (road l2 l7) (road l2 l9) (road l3 l14) (road l3 l5) (road l4 l16) (road l4 l9) (road l5 l17) (road l5 l3) (road l6 l13) (road l6 l8) (road l7 l0) (road l7 l2) (road l8 l15) (road l8 l6) (road l9 l2) (road l9 l4) (spare-in l14) (spare-in l3) (spare-in l4) (vehicle-at l5))
    (:goal (vehicle-at l12))
)