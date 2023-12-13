(define (problem tireworld-023-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l10) (road l0 l20) (road l0 l8) (road l1 l22) (road l1 l5) (road l10 l0) (road l10 l16) (road l10 l17) (road l10 l7) (road l11 l18) (road l11 l3) (road l12 l3) (road l12 l6) (road l13 l20) (road l14 l21) (road l14 l9) (road l15 l3) (road l15 l6) (road l16 l10) (road l16 l5) (road l17 l10) (road l17 l6) (road l18 l11) (road l18 l9) (road l19 l2) (road l19 l8) (road l2 l19) (road l2 l21) (road l20 l0) (road l20 l13) (road l21 l14) (road l21 l2) (road l22 l1) (road l22 l6) (road l3 l11) (road l3 l12) (road l3 l15) (road l4 l6) (road l4 l9) (road l5 l1) (road l5 l16) (road l6 l12) (road l6 l15) (road l6 l17) (road l6 l22) (road l6 l4) (road l6 l7) (road l7 l10) (road l7 l6) (road l8 l0) (road l8 l19) (road l9 l14) (road l9 l18) (road l9 l4) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l5) (spare-in l6) (spare-in l8) (vehicle-at l13))
    (:goal (vehicle-at l21))
)