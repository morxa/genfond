(define (problem tireworld-023-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l14) (road l0 l19) (road l1 l3) (road l1 l9) (road l10 l4) (road l10 l5) (road l11 l2) (road l11 l3) (road l12 l16) (road l12 l8) (road l13 l22) (road l14 l0) (road l14 l16) (road l15 l19) (road l15 l21) (road l16 l12) (road l16 l14) (road l17 l20) (road l17 l8) (road l18 l20) (road l18 l7) (road l19 l0) (road l19 l15) (road l2 l11) (road l2 l4) (road l2 l6) (road l2 l7) (road l20 l17) (road l20 l18) (road l20 l5) (road l21 l15) (road l22 l13) (road l22 l9) (road l3 l1) (road l3 l11) (road l4 l10) (road l4 l2) (road l5 l10) (road l5 l20) (road l5 l6) (road l6 l2) (road l6 l5) (road l7 l18) (road l7 l2) (road l8 l12) (road l8 l17) (road l9 l1) (road l9 l22) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l3) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l21))
    (:goal (vehicle-at l13))
)