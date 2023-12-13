(define (problem tireworld-035-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l19) (road l0 l30) (road l1 l11) (road l1 l5) (road l10 l14) (road l10 l18) (road l10 l34) (road l11 l1) (road l11 l24) (road l11 l31) (road l12 l30) (road l12 l31) (road l13 l15) (road l13 l19) (road l14 l10) (road l14 l28) (road l15 l13) (road l16 l17) (road l16 l20) (road l16 l33) (road l17 l16) (road l17 l7) (road l18 l10) (road l18 l8) (road l19 l0) (road l19 l13) (road l2 l30) (road l2 l33) (road l20 l16) (road l20 l25) (road l21 l24) (road l21 l33) (road l22 l3) (road l23 l27) (road l23 l7) (road l24 l11) (road l24 l21) (road l25 l20) (road l25 l28) (road l26 l6) (road l26 l9) (road l27 l23) (road l27 l30) (road l28 l14) (road l28 l25) (road l29 l32) (road l29 l7) (road l3 l22) (road l3 l32) (road l30 l0) (road l30 l12) (road l30 l2) (road l30 l27) (road l30 l4) (road l31 l11) (road l31 l12) (road l32 l29) (road l32 l3) (road l33 l16) (road l33 l2) (road l33 l21) (road l34 l10) (road l34 l4) (road l4 l30) (road l4 l34) (road l5 l1) (road l5 l9) (road l6 l26) (road l6 l8) (road l7 l17) (road l7 l23) (road l7 l29) (road l8 l18) (road l8 l6) (road l9 l26) (road l9 l5) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l24) (spare-in l25) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l32) (spare-in l34) (spare-in l4) (spare-in l7) (vehicle-at l22))
    (:goal (vehicle-at l15))
)