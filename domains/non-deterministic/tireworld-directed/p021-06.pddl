(define (problem tireworld-021-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l16) (road l0 l3) (road l1 l8) (road l10 l12) (road l10 l14) (road l11 l7) (road l12 l10) (road l12 l17) (road l12 l20) (road l13 l18) (road l14 l10) (road l15 l16) (road l15 l17) (road l16 l0) (road l16 l13) (road l17 l12) (road l17 l15) (road l18 l13) (road l18 l2) (road l18 l4) (road l19 l6) (road l2 l11) (road l20 l12) (road l20 l3) (road l3 l0) (road l3 l20) (road l4 l18) (road l5 l1) (road l5 l9) (road l6 l14) (road l6 l19) (road l7 l11) (road l7 l19) (road l8 l0) (road l9 l5) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l3) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l9))
    (:goal (vehicle-at l4))
)