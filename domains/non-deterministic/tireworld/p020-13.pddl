(define (problem tireworld-020-13)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l1 l10) (road l1 l2) (road l10 l1) (road l10 l11) (road l10 l12) (road l10 l9) (road l11 l10) (road l11 l12) (road l12 l10) (road l12 l11) (road l12 l13) (road l12 l16) (road l13 l12) (road l13 l14) (road l13 l7) (road l13 l8) (road l14 l13) (road l14 l15) (road l14 l2) (road l14 l8) (road l15 l14) (road l16 l12) (road l16 l17) (road l16 l5) (road l17 l16) (road l18 l19) (road l19 l18) (road l19 l5) (road l2 l1) (road l2 l14) (road l2 l3) (road l3 l2) (road l4 l5) (road l5 l16) (road l5 l19) (road l5 l4) (road l5 l6) (road l6 l5) (road l6 l7) (road l7 l13) (road l7 l6) (road l7 l8) (road l8 l13) (road l8 l14) (road l8 l7) (road l8 l9) (road l9 l10) (road l9 l8) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l2) (spare-in l5) (spare-in l8) (vehicle-at l19))
    (:goal (vehicle-at l7))
)