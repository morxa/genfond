(define (problem tireworld-033-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l23) (road l0 l6) (road l1 l9) (road l10 l26) (road l11 l30) (road l11 l32) (road l12 l26) (road l13 l12) (road l13 l17) (road l13 l18) (road l13 l24) (road l14 l4) (road l15 l19) (road l15 l31) (road l16 l2) (road l16 l27) (road l16 l4) (road l17 l13) (road l17 l25) (road l17 l27) (road l18 l13) (road l19 l15) (road l19 l8) (road l2 l16) (road l2 l28) (road l20 l5) (road l20 l9) (road l21 l12) (road l22 l3) (road l22 l30) (road l23 l32) (road l24 l13) (road l24 l26) (road l25 l17) (road l26 l10) (road l27 l16) (road l27 l17) (road l27 l31) (road l28 l2) (road l28 l21) (road l29 l25) (road l29 l30) (road l3 l18) (road l3 l22) (road l3 l5) (road l30 l11) (road l30 l22) (road l30 l29) (road l31 l15) (road l31 l27) (road l32 l11) (road l32 l7) (road l4 l14) (road l4 l16) (road l5 l3) (road l6 l0) (road l6 l8) (road l7 l1) (road l7 l32) (road l8 l6) (road l9 l1) (road l9 l20) (spare-in l12) (spare-in l13) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l26) (spare-in l27) (spare-in l3) (spare-in l30) (spare-in l4) (vehicle-at l14))
    (:goal (vehicle-at l10))
)