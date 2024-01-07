(define (problem tireworld-019-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l10) (road l0 l14) (road l1 l15) (road l1 l5) (road l10 l18) (road l11 l12) (road l11 l15) (road l11 l2) (road l11 l7) (road l12 l11) (road l12 l9) (road l13 l3) (road l14 l0) (road l14 l13) (road l15 l1) (road l15 l11) (road l16 l17) (road l16 l2) (road l16 l3) (road l16 l4) (road l17 l6) (road l18 l10) (road l18 l16) (road l2 l11) (road l3 l13) (road l3 l16) (road l4 l16) (road l5 l1) (road l5 l8) (road l6 l17) (road l7 l11) (road l7 l14) (road l8 l4) (road l8 l5) (road l9 l12) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l3) (spare-in l7) (vehicle-at l9))
    (:goal (vehicle-at l6))
)