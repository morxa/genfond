(define (problem tireworld-033-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l2) (road l0 l20) (road l1 l15) (road l1 l28) (road l1 l29) (road l10 l25) (road l10 l7) (road l11 l13) (road l11 l16) (road l12 l21) (road l13 l11) (road l13 l8) (road l14 l22) (road l14 l31) (road l15 l1) (road l16 l11) (road l16 l23) (road l16 l3) (road l17 l9) (road l18 l28) (road l18 l8) (road l19 l12) (road l19 l22) (road l2 l0) (road l2 l8) (road l20 l0) (road l20 l23) (road l21 l12) (road l21 l15) (road l22 l14) (road l22 l19) (road l22 l27) (road l23 l16) (road l24 l3) (road l24 l31) (road l25 l10) (road l26 l4) (road l26 l6) (road l27 l22) (road l28 l1) (road l28 l18) (road l29 l1) (road l29 l30) (road l3 l16) (road l3 l24) (road l30 l29) (road l30 l5) (road l31 l14) (road l31 l24) (road l32 l31) (road l32 l9) (road l4 l25) (road l4 l26) (road l5 l30) (road l5 l6) (road l6 l26) (road l7 l10) (road l7 l27) (road l8 l13) (road l8 l16) (road l8 l18) (road l8 l2) (road l9 l32) (spare-in l14) (spare-in l24) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l31) (spare-in l32) (spare-in l9) (vehicle-at l17))
    (:goal (vehicle-at l16))
)