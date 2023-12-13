(define (problem tireworld-025-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l0 l15) (road l1 l20) (road l1 l23) (road l10 l20) (road l10 l22) (road l11 l21) (road l11 l7) (road l12 l0) (road l12 l5) (road l13 l20) (road l13 l3) (road l14 l15) (road l14 l17) (road l15 l0) (road l15 l14) (road l15 l24) (road l16 l23) (road l16 l4) (road l17 l14) (road l17 l2) (road l18 l3) (road l19 l22) (road l19 l7) (road l2 l17) (road l2 l6) (road l20 l1) (road l20 l10) (road l20 l13) (road l20 l6) (road l21 l11) (road l21 l3) (road l21 l8) (road l22 l10) (road l22 l19) (road l23 l1) (road l23 l16) (road l23 l5) (road l23 l9) (road l24 l15) (road l24 l8) (road l3 l13) (road l3 l18) (road l3 l21) (road l3 l9) (road l4 l16) (road l4 l5) (road l5 l12) (road l5 l23) (road l5 l4) (road l6 l2) (road l6 l20) (road l7 l11) (road l7 l19) (road l8 l21) (road l8 l24) (road l9 l23) (road l9 l3) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l3) (spare-in l6) (spare-in l7) (spare-in l9) (vehicle-at l8))
    (:goal (vehicle-at l18))
)