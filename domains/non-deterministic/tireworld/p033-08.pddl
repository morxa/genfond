(define (problem tireworld-033-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l22) (road l0 l7) (road l1 l29) (road l1 l6) (road l10 l6) (road l11 l14) (road l11 l24) (road l12 l15) (road l12 l32) (road l13 l19) (road l13 l2) (road l14 l11) (road l14 l27) (road l15 l12) (road l15 l7) (road l16 l25) (road l16 l6) (road l17 l28) (road l17 l4) (road l18 l21) (road l18 l27) (road l19 l13) (road l19 l22) (road l2 l13) (road l2 l23) (road l20 l25) (road l20 l30) (road l21 l18) (road l21 l4) (road l22 l0) (road l22 l19) (road l23 l2) (road l23 l26) (road l24 l11) (road l24 l32) (road l24 l5) (road l25 l16) (road l25 l20) (road l26 l23) (road l26 l9) (road l27 l14) (road l27 l18) (road l28 l17) (road l28 l31) (road l29 l1) (road l29 l31) (road l3 l8) (road l30 l20) (road l30 l9) (road l31 l28) (road l31 l29) (road l32 l12) (road l32 l24) (road l4 l17) (road l4 l21) (road l5 l24) (road l5 l8) (road l6 l1) (road l6 l10) (road l6 l16) (road l7 l0) (road l7 l15) (road l8 l3) (road l8 l5) (road l9 l26) (road l9 l30) (spare-in l0) (spare-in l1) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l3))
    (:goal (vehicle-at l10))
)