(define (problem tireworld-020-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l19) (road l0 l2) (road l0 l9) (road l1 l13) (road l1 l7) (road l10 l3) (road l10 l8) (road l11 l13) (road l11 l16) (road l12 l2) (road l12 l5) (road l13 l1) (road l13 l11) (road l14 l18) (road l14 l7) (road l15 l18) (road l16 l11) (road l16 l17) (road l17 l16) (road l17 l19) (road l18 l14) (road l18 l15) (road l19 l0) (road l19 l17) (road l2 l0) (road l2 l12) (road l2 l4) (road l3 l10) (road l4 l2) (road l4 l8) (road l4 l9) (road l5 l12) (road l5 l7) (road l6 l7) (road l6 l9) (road l7 l1) (road l7 l14) (road l7 l5) (road l7 l6) (road l8 l10) (road l8 l4) (road l9 l0) (road l9 l4) (road l9 l6) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l4) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l3))
    (:goal (vehicle-at l15))
)