(define (problem tireworld-037-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l34) (road l1 l22) (road l10 l2) (road l10 l29) (road l11 l15) (road l11 l24) (road l12 l35) (road l12 l36) (road l13 l28) (road l13 l30) (road l14 l16) (road l14 l22) (road l15 l11) (road l15 l35) (road l16 l14) (road l16 l27) (road l17 l20) (road l17 l32) (road l18 l26) (road l18 l8) (road l19 l31) (road l19 l9) (road l2 l10) (road l2 l4) (road l20 l17) (road l20 l21) (road l21 l20) (road l21 l5) (road l22 l1) (road l22 l14) (road l23 l27) (road l23 l33) (road l24 l11) (road l24 l3) (road l25 l6) (road l25 l8) (road l26 l18) (road l26 l7) (road l27 l16) (road l27 l23) (road l28 l13) (road l28 l34) (road l29 l10) (road l29 l36) (road l3 l24) (road l3 l7) (road l30 l13) (road l30 l9) (road l31 l19) (road l31 l4) (road l32 l17) (road l32 l33) (road l33 l23) (road l33 l32) (road l34 l0) (road l34 l28) (road l35 l12) (road l35 l15) (road l36 l12) (road l36 l29) (road l4 l2) (road l4 l31) (road l5 l21) (road l5 l6) (road l6 l25) (road l6 l5) (road l7 l26) (road l7 l3) (road l8 l18) (road l8 l25) (road l9 l19) (road l9 l30) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l1))
    (:goal (vehicle-at l0))
)