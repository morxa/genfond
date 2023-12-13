(define (problem tireworld-019-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l1 l0) (road l1 l13) (road l1 l2) (road l1 l3) (road l10 l14) (road l10 l16) (road l10 l9) (road l11 l12) (road l12 l11) (road l13 l1) (road l13 l14) (road l13 l5) (road l14 l10) (road l14 l13) (road l14 l15) (road l14 l9) (road l15 l14) (road l15 l16) (road l15 l4) (road l16 l10) (road l16 l15) (road l16 l17) (road l17 l16) (road l17 l18) (road l17 l7) (road l18 l17) (road l2 l1) (road l2 l3) (road l3 l1) (road l3 l2) (road l3 l4) (road l3 l7) (road l4 l15) (road l4 l3) (road l4 l8) (road l5 l13) (road l5 l6) (road l6 l5) (road l6 l8) (road l7 l17) (road l7 l3) (road l7 l8) (road l8 l4) (road l8 l6) (road l8 l7) (road l8 l9) (road l9 l10) (road l9 l14) (road l9 l8) (spare-in l1) (spare-in l10) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l18))
    (:goal (vehicle-at l9))
)