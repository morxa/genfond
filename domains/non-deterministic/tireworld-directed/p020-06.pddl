(define (problem tireworld-020-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l19) (road l0 l8) (road l1 l16) (road l1 l17) (road l10 l15) (road l11 l18) (road l11 l3) (road l12 l16) (road l12 l2) (road l13 l5) (road l14 l7) (road l14 l9) (road l15 l10) (road l15 l2) (road l15 l6) (road l16 l1) (road l16 l12) (road l16 l7) (road l17 l1) (road l17 l4) (road l18 l11) (road l18 l9) (road l19 l0) (road l19 l5) (road l2 l12) (road l2 l15) (road l3 l11) (road l3 l6) (road l4 l17) (road l4 l8) (road l5 l13) (road l5 l19) (road l6 l15) (road l6 l3) (road l7 l14) (road l7 l16) (road l8 l0) (road l8 l4) (road l9 l14) (road l9 l18) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l13))
    (:goal (vehicle-at l10))
)