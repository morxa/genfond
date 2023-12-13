(define (problem tireworld-033-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l24) (road l0 l30) (road l1 l11) (road l1 l30) (road l10 l5) (road l10 l6) (road l11 l1) (road l11 l7) (road l12 l25) (road l12 l4) (road l13 l14) (road l13 l28) (road l14 l13) (road l14 l31) (road l15 l17) (road l16 l19) (road l16 l28) (road l17 l15) (road l17 l3) (road l18 l22) (road l18 l29) (road l19 l16) (road l19 l9) (road l2 l27) (road l2 l7) (road l20 l23) (road l20 l30) (road l21 l29) (road l21 l31) (road l22 l18) (road l22 l9) (road l23 l20) (road l23 l26) (road l24 l0) (road l24 l6) (road l25 l12) (road l25 l5) (road l26 l23) (road l26 l8) (road l27 l2) (road l27 l29) (road l28 l13) (road l28 l16) (road l29 l18) (road l29 l21) (road l29 l27) (road l3 l17) (road l3 l4) (road l30 l0) (road l30 l1) (road l30 l20) (road l31 l14) (road l31 l21) (road l31 l7) (road l32 l7) (road l4 l12) (road l4 l3) (road l5 l10) (road l5 l25) (road l6 l10) (road l6 l24) (road l7 l11) (road l7 l2) (road l7 l31) (road l7 l32) (road l8 l26) (road l8 l9) (road l9 l19) (road l9 l22) (road l9 l8) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l28) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l32))
    (:goal (vehicle-at l15))
)