(define (problem tireworld-032-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l15) (road l0 l31) (road l1 l21) (road l1 l6) (road l10 l27) (road l10 l4) (road l10 l7) (road l11 l22) (road l11 l5) (road l12 l29) (road l12 l5) (road l13 l25) (road l13 l29) (road l13 l8) (road l14 l20) (road l14 l23) (road l15 l0) (road l15 l3) (road l16 l18) (road l16 l9) (road l17 l22) (road l17 l23) (road l18 l16) (road l18 l19) (road l18 l2) (road l18 l20) (road l19 l18) (road l19 l24) (road l2 l18) (road l2 l27) (road l20 l14) (road l20 l18) (road l20 l30) (road l21 l1) (road l21 l26) (road l21 l9) (road l22 l11) (road l22 l17) (road l23 l14) (road l23 l17) (road l24 l19) (road l24 l27) (road l25 l13) (road l25 l3) (road l26 l21) (road l26 l7) (road l27 l10) (road l27 l2) (road l27 l24) (road l27 l7) (road l28 l4) (road l29 l12) (road l29 l13) (road l3 l15) (road l3 l25) (road l30 l20) (road l30 l7) (road l31 l0) (road l31 l4) (road l4 l10) (road l4 l28) (road l4 l31) (road l5 l11) (road l5 l12) (road l6 l1) (road l6 l8) (road l7 l10) (road l7 l26) (road l7 l27) (road l7 l30) (road l8 l13) (road l8 l6) (road l9 l16) (road l9 l21) (spare-in l1) (spare-in l10) (spare-in l16) (spare-in l18) (spare-in l20) (spare-in l21) (spare-in l30) (spare-in l4) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l28))
    (:goal (vehicle-at l13))
)