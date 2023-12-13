(define (problem tireworld-021-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l6) (road l0 l8) (road l1 l14) (road l1 l18) (road l10 l13) (road l10 l15) (road l11 l15) (road l11 l8) (road l12 l2) (road l12 l3) (road l13 l10) (road l13 l17) (road l13 l6) (road l14 l1) (road l14 l16) (road l15 l10) (road l15 l11) (road l15 l9) (road l16 l14) (road l16 l7) (road l17 l13) (road l17 l20) (road l18 l1) (road l18 l9) (road l19 l20) (road l19 l4) (road l2 l12) (road l2 l4) (road l20 l17) (road l20 l19) (road l3 l12) (road l3 l7) (road l4 l19) (road l4 l2) (road l5 l6) (road l5 l9) (road l6 l0) (road l6 l13) (road l6 l5) (road l7 l16) (road l7 l3) (road l8 l0) (road l8 l11) (road l9 l15) (road l9 l18) (road l9 l5) (spare-in l0) (spare-in l1) (spare-in l14) (spare-in l18) (spare-in l19) (spare-in l4) (spare-in l9) (vehicle-at l15))
    (:goal (vehicle-at l16))
)