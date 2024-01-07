(define (problem tireworld-036-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l15) (road l1 l8) (road l10 l24) (road l11 l18) (road l12 l22) (road l12 l33) (road l13 l16) (road l13 l3) (road l13 l6) (road l14 l28) (road l15 l3) (road l16 l14) (road l17 l1) (road l18 l17) (road l19 l10) (road l2 l35) (road l20 l2) (road l20 l26) (road l21 l34) (road l22 l21) (road l23 l29) (road l24 l23) (road l25 l33) (road l26 l19) (road l27 l30) (road l28 l11) (road l29 l31) (road l3 l13) (road l30 l27) (road l30 l32) (road l31 l4) (road l32 l9) (road l33 l12) (road l34 l2) (road l35 l29) (road l4 l0) (road l5 l27) (road l6 l25) (road l7 l20) (road l8 l5) (road l9 l7) (spare-in l0) (spare-in l12) (spare-in l13) (spare-in l16) (spare-in l17) (spare-in l21) (spare-in l22) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l30) (spare-in l32) (spare-in l35) (vehicle-at l3))
    (:goal (vehicle-at l14))
)