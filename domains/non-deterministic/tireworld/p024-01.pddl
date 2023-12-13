(define (problem tireworld-024-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l13) (road l1 l0) (road l1 l15) (road l10 l18) (road l10 l7) (road l11 l12) (road l11 l3) (road l12 l11) (road l12 l18) (road l13 l0) (road l13 l4) (road l14 l16) (road l14 l8) (road l15 l1) (road l15 l5) (road l16 l14) (road l16 l6) (road l17 l21) (road l17 l7) (road l18 l10) (road l18 l12) (road l19 l20) (road l19 l9) (road l2 l22) (road l2 l23) (road l20 l19) (road l20 l4) (road l21 l17) (road l21 l6) (road l22 l2) (road l22 l23) (road l22 l5) (road l23 l2) (road l23 l22) (road l23 l8) (road l3 l11) (road l3 l9) (road l4 l13) (road l4 l20) (road l5 l15) (road l5 l22) (road l6 l16) (road l6 l21) (road l7 l10) (road l7 l17) (road l8 l14) (road l8 l23) (road l9 l19) (road l9 l3) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l9) (vehicle-at l22))
    (:goal (vehicle-at l23))
)