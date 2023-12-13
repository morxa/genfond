(define (problem tireworld-020-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l0 l4) (road l1 l3) (road l1 l6) (road l10 l18) (road l10 l5) (road l11 l19) (road l11 l2) (road l11 l6) (road l12 l0) (road l12 l2) (road l12 l8) (road l13 l17) (road l13 l7) (road l14 l19) (road l14 l5) (road l15 l16) (road l15 l18) (road l16 l15) (road l16 l7) (road l17 l13) (road l17 l3) (road l18 l10) (road l18 l15) (road l19 l11) (road l19 l14) (road l2 l11) (road l2 l12) (road l3 l1) (road l3 l17) (road l3 l8) (road l4 l0) (road l4 l9) (road l5 l10) (road l5 l14) (road l6 l1) (road l6 l11) (road l7 l13) (road l7 l16) (road l8 l12) (road l8 l3) (road l9 l4) (spare-in l0) (spare-in l10) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l8) (vehicle-at l9))
    (:goal (vehicle-at l13))
)