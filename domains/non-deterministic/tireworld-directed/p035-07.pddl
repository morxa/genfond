(define (problem tireworld-035-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l16) (road l0 l3) (road l1 l11) (road l1 l14) (road l10 l26) (road l10 l34) (road l10 l8) (road l11 l1) (road l11 l12) (road l11 l3) (road l12 l13) (road l13 l31) (road l14 l1) (road l14 l5) (road l15 l23) (road l15 l28) (road l16 l0) (road l16 l25) (road l16 l32) (road l17 l28) (road l17 l33) (road l18 l25) (road l19 l22) (road l19 l6) (road l2 l29) (road l2 l31) (road l20 l30) (road l20 l7) (road l21 l5) (road l22 l19) (road l22 l9) (road l23 l15) (road l23 l18) (road l24 l27) (road l25 l16) (road l25 l18) (road l26 l10) (road l26 l33) (road l27 l24) (road l27 l8) (road l28 l15) (road l28 l17) (road l29 l2) (road l29 l24) (road l3 l0) (road l3 l11) (road l30 l20) (road l30 l4) (road l31 l13) (road l31 l2) (road l32 l16) (road l32 l9) (road l33 l17) (road l34 l10) (road l4 l30) (road l5 l14) (road l6 l19) (road l6 l21) (road l7 l20) (road l7 l9) (road l8 l10) (road l8 l27) (road l9 l22) (road l9 l32) (road l9 l7) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l4))
    (:goal (vehicle-at l34))
)