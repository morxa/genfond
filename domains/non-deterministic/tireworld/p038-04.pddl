(define (problem tireworld-038-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l27) (road l0 l33) (road l1 l30) (road l1 l6) (road l10 l27) (road l10 l36) (road l11 l19) (road l11 l21) (road l12 l28) (road l12 l8) (road l13 l20) (road l13 l28) (road l14 l31) (road l14 l37) (road l15 l2) (road l15 l9) (road l16 l26) (road l16 l29) (road l17 l24) (road l17 l31) (road l18 l32) (road l19 l11) (road l19 l6) (road l2 l15) (road l2 l23) (road l20 l13) (road l20 l21) (road l21 l11) (road l21 l20) (road l22 l24) (road l22 l33) (road l23 l2) (road l23 l3) (road l24 l17) (road l24 l22) (road l25 l31) (road l25 l7) (road l26 l16) (road l26 l4) (road l27 l0) (road l27 l10) (road l28 l12) (road l28 l13) (road l29 l16) (road l29 l7) (road l3 l23) (road l3 l30) (road l3 l7) (road l30 l1) (road l30 l3) (road l31 l14) (road l31 l17) (road l31 l25) (road l31 l4) (road l31 l5) (road l32 l18) (road l32 l37) (road l33 l0) (road l33 l22) (road l34 l36) (road l34 l9) (road l35 l5) (road l35 l7) (road l36 l10) (road l36 l34) (road l37 l14) (road l37 l32) (road l4 l26) (road l4 l31) (road l5 l31) (road l5 l35) (road l6 l1) (road l6 l19) (road l7 l25) (road l7 l29) (road l7 l3) (road l7 l35) (road l8 l12) (road l9 l15) (road l9 l34) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l9) (vehicle-at l18))
    (:goal (vehicle-at l8))
)