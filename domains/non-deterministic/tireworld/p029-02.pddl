(define (problem tireworld-029-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l11) (road l0 l5) (road l1 l22) (road l1 l5) (road l10 l24) (road l10 l28) (road l11 l0) (road l11 l9) (road l12 l18) (road l13 l2) (road l13 l21) (road l14 l17) (road l14 l26) (road l15 l18) (road l15 l27) (road l16 l22) (road l16 l4) (road l17 l14) (road l18 l12) (road l18 l15) (road l19 l25) (road l19 l7) (road l19 l9) (road l2 l13) (road l2 l23) (road l20 l4) (road l20 l8) (road l21 l13) (road l21 l26) (road l22 l1) (road l22 l16) (road l23 l2) (road l23 l27) (road l23 l6) (road l24 l10) (road l24 l27) (road l25 l19) (road l25 l3) (road l26 l14) (road l26 l21) (road l26 l8) (road l27 l15) (road l27 l23) (road l27 l24) (road l28 l10) (road l28 l3) (road l3 l25) (road l3 l28) (road l4 l16) (road l4 l20) (road l5 l0) (road l5 l1) (road l6 l23) (road l6 l7) (road l7 l19) (road l7 l6) (road l8 l20) (road l8 l26) (road l9 l11) (road l9 l19) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l12))
    (:goal (vehicle-at l17))
)