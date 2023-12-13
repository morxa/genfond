(define (problem tireworld-032-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l18) (road l0 l21) (road l0 l27) (road l1 l31) (road l1 l4) (road l10 l2) (road l10 l20) (road l11 l5) (road l12 l28) (road l12 l5) (road l13 l14) (road l13 l2) (road l14 l13) (road l14 l15) (road l15 l14) (road l15 l25) (road l15 l26) (road l15 l27) (road l15 l9) (road l16 l22) (road l16 l29) (road l17 l18) (road l17 l25) (road l18 l0) (road l18 l17) (road l18 l24) (road l19 l23) (road l19 l6) (road l19 l8) (road l2 l10) (road l2 l13) (road l2 l22) (road l20 l10) (road l20 l24) (road l21 l0) (road l21 l31) (road l22 l16) (road l22 l2) (road l23 l19) (road l23 l4) (road l24 l18) (road l24 l20) (road l25 l15) (road l25 l17) (road l26 l15) (road l26 l3) (road l27 l0) (road l27 l15) (road l28 l12) (road l28 l7) (road l28 l8) (road l29 l16) (road l29 l7) (road l3 l26) (road l3 l30) (road l30 l3) (road l30 l6) (road l31 l1) (road l31 l21) (road l4 l1) (road l4 l23) (road l5 l11) (road l5 l12) (road l6 l19) (road l6 l30) (road l7 l28) (road l7 l29) (road l7 l9) (road l8 l19) (road l8 l28) (road l9 l15) (road l9 l7) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l27) (spare-in l28) (spare-in l3) (spare-in l31) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l11))
    (:goal (vehicle-at l15))
)