(define (problem tireworld-032-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l20) (road l0 l24) (road l1 l18) (road l10 l11) (road l11 l10) (road l11 l21) (road l12 l28) (road l13 l7) (road l14 l8) (road l14 l9) (road l15 l26) (road l17 l6) (road l18 l2) (road l19 l20) (road l2 l18) (road l2 l3) (road l20 l0) (road l21 l11) (road l21 l31) (road l22 l10) (road l22 l17) (road l23 l1) (road l23 l26) (road l24 l0) (road l24 l22) (road l25 l7) (road l25 l9) (road l26 l15) (road l26 l23) (road l27 l30) (road l27 l9) (road l28 l29) (road l29 l5) (road l3 l19) (road l3 l2) (road l30 l16) (road l30 l27) (road l31 l13) (road l31 l21) (road l4 l8) (road l5 l15) (road l6 l4) (road l7 l25) (road l8 l14) (road l9 l27) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l12))
    (:goal (vehicle-at l16))
)