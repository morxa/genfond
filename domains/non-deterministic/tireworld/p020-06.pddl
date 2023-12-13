(define (problem tireworld-020-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l0 l8) (road l1 l10) (road l1 l13) (road l1 l3) (road l10 l1) (road l10 l4) (road l11 l12) (road l11 l5) (road l12 l0) (road l12 l11) (road l12 l2) (road l12 l4) (road l12 l8) (road l13 l1) (road l13 l15) (road l13 l16) (road l14 l19) (road l15 l13) (road l15 l17) (road l15 l8) (road l16 l13) (road l16 l9) (road l17 l15) (road l17 l4) (road l17 l6) (road l18 l19) (road l18 l7) (road l19 l14) (road l19 l18) (road l19 l5) (road l2 l12) (road l2 l7) (road l3 l1) (road l3 l4) (road l4 l10) (road l4 l12) (road l4 l17) (road l4 l3) (road l5 l11) (road l5 l19) (road l6 l17) (road l6 l8) (road l7 l18) (road l7 l2) (road l8 l0) (road l8 l12) (road l8 l15) (road l8 l6) (road l8 l9) (road l9 l16) (road l9 l8) (spare-in l12) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l6) (spare-in l7) (vehicle-at l14))
    (:goal (vehicle-at l8))
)