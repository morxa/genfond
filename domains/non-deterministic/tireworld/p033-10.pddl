(define (problem tireworld-033-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l11) (road l0 l3) (road l0 l4) (road l1 l25) (road l1 l28) (road l10 l19) (road l10 l5) (road l11 l0) (road l11 l8) (road l12 l23) (road l12 l32) (road l13 l20) (road l13 l28) (road l14 l27) (road l14 l30) (road l14 l7) (road l15 l23) (road l15 l31) (road l16 l3) (road l16 l30) (road l17 l19) (road l17 l29) (road l18 l29) (road l18 l9) (road l19 l10) (road l19 l17) (road l2 l22) (road l2 l24) (road l2 l9) (road l20 l13) (road l20 l32) (road l21 l5) (road l21 l6) (road l22 l2) (road l22 l27) (road l23 l12) (road l23 l15) (road l24 l2) (road l24 l7) (road l25 l1) (road l25 l4) (road l26 l31) (road l26 l6) (road l27 l14) (road l27 l22) (road l28 l1) (road l28 l13) (road l29 l17) (road l29 l18) (road l3 l0) (road l3 l16) (road l30 l14) (road l30 l16) (road l31 l15) (road l31 l26) (road l32 l12) (road l32 l20) (road l4 l0) (road l4 l25) (road l5 l10) (road l5 l21) (road l6 l21) (road l6 l26) (road l7 l14) (road l7 l24) (road l8 l11) (road l9 l18) (road l9 l2) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l9) (vehicle-at l32))
    (:goal (vehicle-at l8))
)