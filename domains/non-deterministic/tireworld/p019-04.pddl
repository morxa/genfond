(define (problem tireworld-019-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l15) (road l0 l8) (road l1 l10) (road l1 l3) (road l10 l1) (road l10 l9) (road l11 l13) (road l11 l16) (road l12 l3) (road l12 l5) (road l13 l11) (road l13 l3) (road l13 l4) (road l13 l7) (road l14 l15) (road l14 l18) (road l14 l4) (road l14 l5) (road l15 l0) (road l15 l14) (road l16 l11) (road l16 l18) (road l17 l2) (road l17 l6) (road l18 l14) (road l18 l16) (road l2 l17) (road l2 l9) (road l3 l1) (road l3 l12) (road l3 l13) (road l4 l13) (road l4 l14) (road l5 l12) (road l5 l14) (road l6 l17) (road l6 l8) (road l7 l13) (road l8 l0) (road l8 l6) (road l9 l10) (road l9 l2) (spare-in l11) (spare-in l13) (spare-in l8) (vehicle-at l3))
    (:goal (vehicle-at l7))
)