(define (problem tireworld-036-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l29) (road l1 l0) (road l1 l11) (road l1 l2) (road l1 l31) (road l1 l8) (road l10 l15) (road l10 l27) (road l11 l1) (road l11 l12) (road l12 l11) (road l12 l16) (road l12 l25) (road l12 l29) (road l12 l30) (road l13 l14) (road l13 l19) (road l14 l13) (road l14 l25) (road l15 l10) (road l15 l9) (road l16 l12) (road l16 l23) (road l16 l7) (road l17 l20) (road l17 l33) (road l18 l22) (road l18 l26) (road l18 l28) (road l19 l13) (road l19 l24) (road l2 l1) (road l2 l5) (road l20 l17) (road l20 l3) (road l21 l27) (road l21 l35) (road l22 l18) (road l23 l16) (road l23 l4) (road l24 l19) (road l24 l32) (road l25 l12) (road l25 l14) (road l26 l18) (road l26 l34) (road l27 l10) (road l27 l21) (road l27 l4) (road l28 l18) (road l28 l35) (road l29 l0) (road l29 l12) (road l3 l20) (road l3 l9) (road l30 l12) (road l31 l1) (road l31 l33) (road l32 l24) (road l32 l9) (road l33 l17) (road l33 l31) (road l33 l6) (road l34 l26) (road l34 l6) (road l35 l21) (road l35 l28) (road l35 l8) (road l4 l23) (road l4 l27) (road l5 l2) (road l5 l7) (road l6 l33) (road l6 l34) (road l7 l16) (road l7 l5) (road l8 l1) (road l8 l35) (road l9 l15) (road l9 l3) (road l9 l32) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l16) (spare-in l18) (spare-in l2) (spare-in l21) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l31) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l9) (vehicle-at l30))
    (:goal (vehicle-at l22))
)