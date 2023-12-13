(define (problem tireworld-020-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l15) (road l0 l19) (road l0 l4) (road l1 l4) (road l1 l9) (road l10 l14) (road l10 l18) (road l11 l17) (road l11 l3) (road l12 l17) (road l13 l2) (road l13 l4) (road l14 l10) (road l14 l6) (road l15 l0) (road l15 l3) (road l16 l19) (road l16 l8) (road l17 l11) (road l17 l12) (road l17 l5) (road l18 l10) (road l19 l0) (road l19 l16) (road l2 l13) (road l2 l5) (road l3 l11) (road l3 l15) (road l4 l0) (road l4 l1) (road l4 l13) (road l5 l17) (road l5 l2) (road l5 l7) (road l6 l14) (road l6 l8) (road l7 l5) (road l7 l9) (road l8 l16) (road l8 l6) (road l9 l1) (road l9 l7) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l3) (spare-in l6) (spare-in l8) (vehicle-at l12))
    (:goal (vehicle-at l18))
)