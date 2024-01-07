(define (problem tireworld-035-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l16) (road l0 l5) (road l1 l19) (road l10 l26) (road l10 l30) (road l11 l13) (road l12 l33) (road l13 l11) (road l13 l30) (road l14 l17) (road l15 l5) (road l16 l12) (road l17 l14) (road l17 l7) (road l18 l29) (road l19 l18) (road l2 l34) (road l2 l6) (road l20 l5) (road l21 l25) (road l22 l27) (road l22 l8) (road l23 l34) (road l24 l31) (road l24 l8) (road l24 l9) (road l25 l14) (road l25 l21) (road l26 l28) (road l26 l31) (road l27 l11) (road l27 l22) (road l28 l4) (road l29 l24) (road l3 l23) (road l3 l4) (road l30 l10) (road l31 l26) (road l32 l20) (road l33 l12) (road l34 l2) (road l34 l23) (road l4 l3) (road l5 l0) (road l5 l15) (road l6 l15) (road l7 l32) (road l8 l24) (road l9 l21) (road l9 l22) (road l9 l24) (spare-in l0) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l34) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l8) (vehicle-at l1))
    (:goal (vehicle-at l33))
)