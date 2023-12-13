(define (problem tireworld-034-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l21) (road l0 l23) (road l1 l27) (road l1 l30) (road l10 l20) (road l10 l30) (road l11 l12) (road l11 l16) (road l12 l11) (road l12 l28) (road l12 l6) (road l13 l22) (road l13 l31) (road l14 l19) (road l14 l31) (road l15 l17) (road l15 l8) (road l16 l11) (road l16 l29) (road l16 l9) (road l17 l15) (road l17 l7) (road l18 l2) (road l18 l23) (road l19 l14) (road l19 l26) (road l2 l18) (road l2 l6) (road l20 l10) (road l21 l0) (road l21 l24) (road l22 l13) (road l22 l5) (road l23 l0) (road l23 l18) (road l24 l21) (road l24 l4) (road l25 l28) (road l25 l5) (road l26 l19) (road l26 l33) (road l27 l1) (road l27 l8) (road l28 l12) (road l28 l25) (road l29 l16) (road l29 l33) (road l3 l32) (road l3 l4) (road l30 l1) (road l30 l10) (road l31 l13) (road l31 l14) (road l32 l3) (road l32 l7) (road l33 l26) (road l33 l29) (road l4 l24) (road l4 l3) (road l5 l22) (road l5 l25) (road l6 l12) (road l6 l2) (road l7 l17) (road l7 l32) (road l8 l15) (road l8 l27) (road l9 l16) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l9))
    (:goal (vehicle-at l20))
)