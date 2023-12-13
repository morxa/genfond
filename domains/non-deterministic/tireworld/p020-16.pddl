(define (problem tireworld-020-16)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l10) (road l1 l0) (road l1 l2) (road l1 l3) (road l10 l0) (road l10 l11) (road l10 l9) (road l11 l10) (road l11 l12) (road l12 l11) (road l13 l14) (road l13 l17) (road l13 l3) (road l14 l13) (road l14 l15) (road l15 l14) (road l16 l17) (road l17 l13) (road l17 l16) (road l17 l18) (road l17 l6) (road l18 l17) (road l18 l19) (road l19 l18) (road l2 l1) (road l2 l3) (road l3 l1) (road l3 l13) (road l3 l2) (road l3 l4) (road l4 l3) (road l5 l6) (road l6 l17) (road l6 l5) (road l6 l7) (road l7 l6) (road l7 l8) (road l8 l7) (road l8 l9) (road l9 l10) (road l9 l8) (spare-in l0) (spare-in l1) (spare-in l13) (spare-in l17) (spare-in l3) (spare-in l6) (vehicle-at l10))
    (:goal (vehicle-at l5))
)