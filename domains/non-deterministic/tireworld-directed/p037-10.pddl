(define (problem tireworld-037-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l5) (road l1 l20) (road l10 l17) (road l12 l11) (road l13 l2) (road l14 l31) (road l15 l9) (road l16 l15) (road l17 l23) (road l18 l26) (road l19 l0) (road l19 l28) (road l19 l8) (road l2 l33) (road l20 l5) (road l21 l3) (road l22 l4) (road l23 l5) (road l24 l22) (road l25 l11) (road l26 l18) (road l26 l32) (road l27 l29) (road l27 l30) (road l27 l5) (road l28 l14) (road l29 l12) (road l3 l10) (road l3 l34) (road l30 l36) (road l31 l16) (road l31 l18) (road l32 l13) (road l33 l24) (road l34 l11) (road l35 l6) (road l36 l21) (road l4 l1) (road l4 l22) (road l5 l20) (road l5 l27) (road l6 l8) (road l7 l35) (road l8 l25) (road l9 l7) (spare-in l25) (spare-in l26) (spare-in l30) (spare-in l35) (spare-in l5) (spare-in l8) (vehicle-at l19))
    (:goal (vehicle-at l11))
)