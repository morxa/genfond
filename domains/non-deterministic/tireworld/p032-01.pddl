(define (problem tireworld-032-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l16) (road l0 l22) (road l0 l6) (road l1 l20) (road l1 l8) (road l10 l22) (road l10 l27) (road l11 l14) (road l11 l2) (road l12 l31) (road l12 l4) (road l13 l4) (road l13 l9) (road l14 l11) (road l14 l29) (road l15 l2) (road l15 l23) (road l16 l0) (road l16 l28) (road l17 l5) (road l18 l21) (road l18 l27) (road l19 l26) (road l19 l7) (road l2 l11) (road l2 l15) (road l20 l1) (road l20 l23) (road l20 l30) (road l21 l18) (road l21 l29) (road l22 l0) (road l22 l10) (road l23 l15) (road l23 l20) (road l24 l3) (road l25 l26) (road l25 l28) (road l26 l19) (road l26 l25) (road l27 l10) (road l27 l18) (road l28 l16) (road l28 l25) (road l29 l14) (road l29 l21) (road l3 l24) (road l3 l8) (road l30 l20) (road l30 l31) (road l31 l12) (road l31 l30) (road l4 l12) (road l4 l13) (road l5 l17) (road l5 l6) (road l6 l0) (road l6 l5) (road l7 l19) (road l7 l9) (road l8 l1) (road l8 l3) (road l9 l13) (road l9 l7) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l14) (spare-in l15) (spare-in l18) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l25) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l5) (spare-in l6) (spare-in l8) (vehicle-at l17))
    (:goal (vehicle-at l24))
)