(define (problem tireworld-027-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l19) (road l0 l23) (road l1 l22) (road l1 l25) (road l10 l18) (road l10 l8) (road l11 l21) (road l11 l4) (road l12 l24) (road l12 l4) (road l13 l15) (road l13 l24) (road l14 l26) (road l14 l3) (road l15 l13) (road l15 l16) (road l16 l15) (road l16 l18) (road l17 l3) (road l17 l9) (road l18 l10) (road l18 l16) (road l19 l0) (road l19 l20) (road l2 l6) (road l2 l7) (road l20 l19) (road l21 l11) (road l21 l9) (road l22 l1) (road l22 l23) (road l23 l0) (road l23 l22) (road l24 l12) (road l24 l13) (road l25 l1) (road l25 l8) (road l26 l14) (road l26 l7) (road l3 l14) (road l3 l17) (road l4 l11) (road l4 l12) (road l5 l6) (road l6 l2) (road l6 l5) (road l7 l2) (road l7 l26) (road l8 l10) (road l8 l25) (road l9 l17) (road l9 l21) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l3) (spare-in l4) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l5))
    (:goal (vehicle-at l20))
)