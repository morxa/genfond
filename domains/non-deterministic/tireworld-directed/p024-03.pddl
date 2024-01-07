(define (problem tireworld-024-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l18) (road l1 l17) (road l10 l7) (road l11 l1) (road l12 l6) (road l13 l17) (road l13 l8) (road l14 l16) (road l14 l2) (road l15 l3) (road l15 l5) (road l15 l7) (road l16 l14) (road l16 l3) (road l17 l13) (road l18 l0) (road l18 l22) (road l19 l3) (road l19 l4) (road l2 l14) (road l2 l5) (road l20 l23) (road l20 l3) (road l21 l4) (road l21 l5) (road l22 l18) (road l23 l20) (road l23 l8) (road l3 l15) (road l3 l16) (road l3 l19) (road l3 l20) (road l4 l10) (road l4 l19) (road l4 l21) (road l5 l15) (road l5 l2) (road l5 l21) (road l5 l9) (road l6 l12) (road l6 l22) (road l7 l10) (road l7 l15) (road l8 l13) (road l8 l23) (road l9 l12) (road l9 l5) (spare-in l1) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l3) (spare-in l5) (spare-in l6) (spare-in l8) (spare-in l9) (vehicle-at l11))
    (:goal (vehicle-at l0))
)