(define (problem tireworld-028-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l21) (road l0 l7) (road l1 l10) (road l1 l5) (road l10 l1) (road l10 l17) (road l10 l7) (road l11 l20) (road l11 l3) (road l12 l16) (road l12 l4) (road l12 l7) (road l13 l17) (road l13 l19) (road l14 l18) (road l14 l8) (road l15 l18) (road l15 l3) (road l16 l12) (road l16 l25) (road l17 l10) (road l17 l13) (road l18 l14) (road l18 l15) (road l19 l13) (road l19 l2) (road l2 l19) (road l2 l20) (road l20 l11) (road l20 l2) (road l21 l0) (road l21 l24) (road l21 l5) (road l22 l23) (road l22 l7) (road l23 l22) (road l23 l9) (road l24 l21) (road l24 l26) (road l25 l16) (road l25 l6) (road l26 l24) (road l26 l27) (road l26 l8) (road l27 l26) (road l27 l9) (road l3 l11) (road l3 l15) (road l4 l12) (road l4 l9) (road l5 l1) (road l5 l21) (road l6 l25) (road l7 l0) (road l7 l10) (road l7 l12) (road l7 l22) (road l8 l14) (road l8 l26) (road l9 l23) (road l9 l27) (road l9 l4) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l7) (spare-in l9) (vehicle-at l10))
    (:goal (vehicle-at l6))
)