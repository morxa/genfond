(define (problem tireworld-032-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l14) (road l0 l22) (road l0 l27) (road l0 l28) (road l1 l19) (road l1 l23) (road l10 l12) (road l10 l6) (road l11 l15) (road l11 l29) (road l12 l10) (road l12 l24) (road l13 l18) (road l13 l23) (road l14 l0) (road l14 l25) (road l14 l8) (road l15 l11) (road l15 l9) (road l16 l25) (road l16 l4) (road l17 l2) (road l17 l3) (road l18 l13) (road l18 l4) (road l19 l1) (road l19 l26) (road l2 l17) (road l2 l9) (road l20 l3) (road l20 l30) (road l21 l27) (road l21 l31) (road l22 l0) (road l22 l7) (road l23 l1) (road l23 l13) (road l24 l12) (road l24 l7) (road l25 l14) (road l25 l16) (road l26 l19) (road l26 l5) (road l27 l0) (road l27 l21) (road l28 l0) (road l29 l11) (road l3 l17) (road l3 l20) (road l30 l20) (road l30 l5) (road l31 l21) (road l31 l7) (road l4 l16) (road l4 l18) (road l5 l26) (road l5 l30) (road l6 l10) (road l6 l8) (road l7 l22) (road l7 l24) (road l7 l31) (road l8 l14) (road l8 l6) (road l9 l15) (road l9 l2) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l29))
    (:goal (vehicle-at l28))
)