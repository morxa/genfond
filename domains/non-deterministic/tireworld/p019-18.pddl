(define (problem tireworld-019-18)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l1 l0) (road l1 l2) (road l10 l11) (road l10 l14) (road l10 l7) (road l11 l10) (road l11 l16) (road l11 l8) (road l13 l14) (road l14 l10) (road l14 l13) (road l14 l15) (road l14 l16) (road l15 l14) (road l15 l16) (road l16 l11) (road l16 l14) (road l16 l15) (road l16 l17) (road l17 l16) (road l17 l18) (road l17 l7) (road l17 l9) (road l18 l17) (road l2 l1) (road l2 l3) (road l3 l2) (road l3 l4) (road l4 l3) (road l4 l5) (road l5 l4) (road l5 l6) (road l6 l5) (road l6 l7) (road l7 l10) (road l7 l17) (road l7 l6) (road l7 l8) (road l8 l11) (road l8 l7) (road l8 l9) (road l9 l17) (road l9 l8) (spare-in l10) (spare-in l11) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l7) (vehicle-at l8))
    (:goal (vehicle-at l9))
)