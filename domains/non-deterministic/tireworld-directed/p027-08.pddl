(define (problem tireworld-027-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l21) (road l0 l9) (road l1 l19) (road l1 l24) (road l10 l26) (road l11 l2) (road l11 l21) (road l12 l2) (road l12 l24) (road l13 l7) (road l13 l8) (road l14 l22) (road l14 l3) (road l15 l19) (road l16 l17) (road l16 l8) (road l17 l16) (road l17 l22) (road l18 l26) (road l18 l6) (road l19 l1) (road l19 l15) (road l2 l11) (road l2 l12) (road l20 l3) (road l20 l5) (road l21 l0) (road l21 l11) (road l21 l4) (road l22 l14) (road l22 l17) (road l23 l1) (road l23 l25) (road l24 l1) (road l24 l12) (road l25 l23) (road l25 l3) (road l26 l10) (road l26 l18) (road l26 l23) (road l3 l14) (road l3 l20) (road l3 l25) (road l4 l10) (road l4 l21) (road l5 l15) (road l5 l20) (road l6 l18) (road l7 l13) (road l8 l13) (road l8 l16) (road l9 l0) (road l9 l7) (spare-in l18) (spare-in l20) (spare-in l25) (spare-in l26) (vehicle-at l6))
    (:goal (vehicle-at l23))
)