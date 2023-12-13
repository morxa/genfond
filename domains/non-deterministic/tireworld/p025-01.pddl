(define (problem tireworld-025-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l13) (road l0 l14) (road l1 l17) (road l1 l5) (road l10 l15) (road l10 l19) (road l10 l2) (road l11 l12) (road l11 l9) (road l12 l11) (road l12 l21) (road l13 l0) (road l13 l19) (road l14 l0) (road l14 l8) (road l15 l10) (road l15 l3) (road l16 l4) (road l16 l5) (road l17 l1) (road l17 l7) (road l18 l2) (road l18 l20) (road l19 l10) (road l19 l13) (road l2 l10) (road l2 l18) (road l20 l18) (road l20 l3) (road l20 l6) (road l21 l12) (road l21 l3) (road l22 l24) (road l22 l7) (road l23 l4) (road l23 l9) (road l24 l22) (road l24 l6) (road l3 l15) (road l3 l20) (road l3 l21) (road l4 l16) (road l4 l23) (road l5 l1) (road l5 l16) (road l6 l20) (road l6 l24) (road l7 l17) (road l7 l22) (road l7 l8) (road l8 l14) (road l8 l7) (road l9 l11) (road l9 l23) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l3))
    (:goal (vehicle-at l19))
)