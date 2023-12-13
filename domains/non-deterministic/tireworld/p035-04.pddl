(define (problem tireworld-035-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l20) (road l0 l23) (road l1 l33) (road l1 l5) (road l10 l24) (road l10 l32) (road l11 l17) (road l11 l18) (road l12 l2) (road l12 l25) (road l13 l21) (road l13 l32) (road l14 l25) (road l14 l8) (road l15 l18) (road l15 l29) (road l16 l20) (road l16 l34) (road l17 l11) (road l17 l24) (road l18 l11) (road l18 l15) (road l18 l33) (road l18 l4) (road l19 l21) (road l19 l28) (road l2 l12) (road l20 l0) (road l20 l16) (road l21 l13) (road l21 l19) (road l22 l30) (road l22 l7) (road l23 l0) (road l23 l7) (road l24 l10) (road l24 l17) (road l25 l12) (road l25 l14) (road l26 l30) (road l26 l9) (road l27 l29) (road l27 l9) (road l28 l19) (road l28 l34) (road l29 l15) (road l29 l27) (road l29 l3) (road l3 l29) (road l3 l8) (road l30 l22) (road l30 l26) (road l31 l32) (road l31 l4) (road l32 l10) (road l32 l13) (road l32 l31) (road l33 l1) (road l33 l18) (road l34 l16) (road l34 l28) (road l4 l18) (road l4 l31) (road l4 l5) (road l5 l1) (road l5 l4) (road l5 l6) (road l6 l5) (road l7 l22) (road l7 l23) (road l8 l14) (road l8 l3) (road l9 l26) (road l9 l27) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l5) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l6))
    (:goal (vehicle-at l2))
)