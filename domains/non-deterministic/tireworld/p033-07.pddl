(define (problem tireworld-033-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l22) (road l0 l28) (road l1 l2) (road l1 l27) (road l10 l20) (road l10 l3) (road l11 l25) (road l11 l26) (road l11 l29) (road l12 l16) (road l12 l28) (road l13 l23) (road l13 l7) (road l14 l25) (road l14 l5) (road l15 l2) (road l15 l9) (road l16 l12) (road l16 l21) (road l17 l23) (road l17 l26) (road l18 l19) (road l18 l24) (road l19 l18) (road l19 l32) (road l2 l1) (road l2 l15) (road l20 l10) (road l20 l26) (road l20 l30) (road l20 l4) (road l21 l16) (road l21 l4) (road l22 l0) (road l22 l6) (road l23 l13) (road l23 l17) (road l24 l18) (road l24 l3) (road l25 l11) (road l25 l14) (road l26 l11) (road l26 l17) (road l26 l20) (road l27 l1) (road l27 l31) (road l28 l0) (road l28 l12) (road l29 l11) (road l29 l9) (road l3 l10) (road l3 l24) (road l3 l5) (road l30 l20) (road l30 l9) (road l31 l27) (road l31 l7) (road l32 l19) (road l4 l20) (road l4 l21) (road l5 l14) (road l5 l3) (road l6 l22) (road l6 l8) (road l7 l13) (road l7 l31) (road l8 l6) (road l9 l15) (road l9 l29) (road l9 l30) (spare-in l0) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l9) (vehicle-at l8))
    (:goal (vehicle-at l32))
)