(define (problem tireworld-037-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l29) (road l0 l36) (road l1 l12) (road l1 l5) (road l10 l31) (road l10 l7) (road l11 l6) (road l11 l8) (road l12 l1) (road l12 l13) (road l12 l15) (road l12 l30) (road l13 l12) (road l13 l23) (road l14 l21) (road l14 l34) (road l15 l12) (road l15 l22) (road l16 l23) (road l16 l28) (road l17 l18) (road l17 l9) (road l18 l17) (road l18 l35) (road l19 l2) (road l19 l6) (road l2 l19) (road l2 l25) (road l20 l34) (road l20 l9) (road l21 l14) (road l21 l3) (road l22 l15) (road l22 l27) (road l23 l13) (road l23 l16) (road l23 l7) (road l24 l30) (road l24 l5) (road l25 l2) (road l25 l36) (road l26 l32) (road l26 l33) (road l27 l22) (road l28 l16) (road l28 l29) (road l29 l0) (road l29 l28) (road l3 l21) (road l3 l4) (road l30 l12) (road l30 l24) (road l31 l10) (road l32 l26) (road l32 l35) (road l33 l26) (road l33 l5) (road l34 l14) (road l34 l20) (road l35 l18) (road l35 l32) (road l36 l0) (road l36 l25) (road l4 l3) (road l4 l8) (road l5 l1) (road l5 l24) (road l5 l33) (road l6 l11) (road l6 l19) (road l7 l10) (road l7 l23) (road l8 l11) (road l8 l4) (road l9 l17) (road l9 l20) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l27))
    (:goal (vehicle-at l31))
)