(define (problem tireworld-038-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l2) (road l0 l31) (road l1 l22) (road l1 l30) (road l10 l3) (road l10 l4) (road l11 l22) (road l11 l6) (road l12 l25) (road l12 l6) (road l13 l29) (road l13 l36) (road l14 l19) (road l14 l9) (road l15 l25) (road l15 l37) (road l16 l21) (road l16 l28) (road l17 l27) (road l18 l3) (road l18 l5) (road l19 l14) (road l19 l31) (road l2 l0) (road l2 l17) (road l20 l34) (road l20 l5) (road l21 l16) (road l21 l27) (road l22 l1) (road l22 l11) (road l22 l29) (road l22 l32) (road l23 l4) (road l24 l32) (road l24 l36) (road l25 l12) (road l25 l15) (road l26 l35) (road l26 l9) (road l27 l17) (road l27 l21) (road l28 l16) (road l28 l36) (road l29 l13) (road l29 l22) (road l3 l10) (road l3 l18) (road l30 l1) (road l30 l8) (road l31 l0) (road l31 l19) (road l31 l7) (road l32 l22) (road l32 l24) (road l33 l37) (road l34 l20) (road l34 l36) (road l35 l26) (road l36 l13) (road l36 l24) (road l36 l28) (road l36 l34) (road l37 l15) (road l37 l33) (road l4 l10) (road l4 l23) (road l5 l20) (road l6 l11) (road l6 l12) (road l6 l7) (road l7 l31) (road l7 l6) (road l8 l30) (road l8 l33) (road l9 l14) (road l9 l26) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l22) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l36) (spare-in l37) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l23))
    (:goal (vehicle-at l35))
)