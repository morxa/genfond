(define (problem tireworld-032-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l14) (road l0 l4) (road l1 l3) (road l1 l8) (road l10 l3) (road l10 l4) (road l11 l15) (road l11 l30) (road l12 l13) (road l12 l19) (road l13 l12) (road l13 l15) (road l13 l9) (road l14 l0) (road l14 l19) (road l15 l11) (road l15 l13) (road l16 l29) (road l16 l4) (road l17 l21) (road l17 l7) (road l17 l8) (road l18 l21) (road l18 l25) (road l19 l12) (road l19 l14) (road l19 l23) (road l2 l28) (road l2 l6) (road l20 l22) (road l20 l27) (road l21 l17) (road l21 l18) (road l22 l20) (road l22 l28) (road l23 l19) (road l23 l24) (road l24 l23) (road l24 l27) (road l25 l18) (road l26 l5) (road l27 l20) (road l27 l24) (road l28 l2) (road l28 l22) (road l29 l16) (road l29 l5) (road l3 l1) (road l3 l10) (road l30 l11) (road l30 l6) (road l31 l7) (road l31 l9) (road l4 l0) (road l4 l10) (road l4 l16) (road l5 l26) (road l5 l29) (road l6 l2) (road l6 l30) (road l7 l17) (road l7 l31) (road l8 l1) (road l8 l17) (road l9 l13) (road l9 l31) (spare-in l1) (spare-in l10) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l20) (spare-in l21) (spare-in l23) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l31) (spare-in l4) (spare-in l5) (spare-in l8) (vehicle-at l25))
    (:goal (vehicle-at l26))
)