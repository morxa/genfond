(define (problem tireworld-035-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l18) (road l0 l32) (road l1 l29) (road l1 l33) (road l10 l14) (road l10 l20) (road l11 l21) (road l11 l26) (road l11 l30) (road l11 l34) (road l11 l6) (road l12 l16) (road l12 l9) (road l13 l28) (road l13 l34) (road l14 l10) (road l14 l6) (road l15 l22) (road l15 l31) (road l16 l12) (road l16 l21) (road l16 l27) (road l16 l3) (road l16 l7) (road l17 l30) (road l17 l7) (road l18 l0) (road l18 l2) (road l19 l22) (road l19 l23) (road l2 l18) (road l2 l24) (road l2 l5) (road l20 l10) (road l20 l23) (road l21 l11) (road l21 l16) (road l22 l15) (road l22 l19) (road l23 l19) (road l23 l20) (road l24 l2) (road l24 l30) (road l25 l33) (road l25 l4) (road l26 l11) (road l26 l4) (road l27 l16) (road l27 l28) (road l28 l13) (road l28 l27) (road l29 l1) (road l29 l8) (road l3 l16) (road l3 l5) (road l30 l11) (road l30 l17) (road l30 l24) (road l31 l15) (road l31 l8) (road l32 l0) (road l32 l9) (road l33 l1) (road l33 l25) (road l34 l11) (road l34 l13) (road l4 l25) (road l4 l26) (road l5 l2) (road l5 l3) (road l6 l11) (road l6 l14) (road l7 l16) (road l7 l17) (road l8 l29) (road l8 l31) (road l9 l12) (road l9 l32) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l4) (spare-in l5) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l6))
    (:goal (vehicle-at l0))
)