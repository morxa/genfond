(define (problem tireworld-020-5)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l18) (road l1 l0) (road l1 l5) (road l10 l11) (road l10 l9) (road l11 l10) (road l11 l12) (road l11 l16) (road l11 l8) (road l12 l11) (road l12 l13) (road l12 l17) (road l12 l9) (road l13 l12) (road l13 l14) (road l14 l13) (road l14 l15) (road l14 l18) (road l15 l14) (road l15 l16) (road l16 l11) (road l16 l15) (road l16 l6) (road l17 l12) (road l17 l18) (road l18 l0) (road l18 l14) (road l18 l17) (road l18 l19) (road l19 l18) (road l19 l4) (road l19 l6) (road l2 l3) (road l3 l2) (road l3 l4) (road l3 l7) (road l3 l8) (road l4 l19) (road l4 l3) (road l4 l5) (road l5 l1) (road l5 l4) (road l6 l16) (road l6 l19) (road l7 l3) (road l7 l8) (road l7 l9) (road l8 l11) (road l8 l3) (road l8 l7) (road l8 l9) (road l9 l10) (road l9 l12) (road l9 l7) (road l9 l8) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l17))
    (:goal (vehicle-at l14))
)