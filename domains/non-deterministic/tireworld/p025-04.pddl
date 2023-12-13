(define (problem tireworld-025-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l2) (road l0 l6) (road l1 l10) (road l1 l17) (road l10 l1) (road l10 l14) (road l11 l21) (road l11 l22) (road l12 l24) (road l12 l5) (road l13 l17) (road l13 l19) (road l13 l20) (road l14 l10) (road l14 l23) (road l15 l19) (road l15 l8) (road l16 l2) (road l16 l4) (road l17 l1) (road l17 l13) (road l18 l4) (road l18 l7) (road l19 l13) (road l19 l15) (road l2 l0) (road l2 l16) (road l2 l8) (road l20 l13) (road l20 l23) (road l21 l11) (road l21 l8) (road l22 l11) (road l22 l3) (road l23 l14) (road l23 l20) (road l23 l7) (road l23 l9) (road l24 l12) (road l24 l3) (road l3 l22) (road l3 l24) (road l4 l16) (road l4 l18) (road l4 l5) (road l5 l12) (road l5 l4) (road l6 l0) (road l6 l9) (road l7 l18) (road l7 l23) (road l8 l15) (road l8 l2) (road l8 l21) (road l9 l23) (road l9 l6) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l16) (spare-in l18) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l2))
    (:goal (vehicle-at l21))
)