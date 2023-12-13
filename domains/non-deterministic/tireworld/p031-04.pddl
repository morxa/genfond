(define (problem tireworld-031-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l25) (road l0 l28) (road l1 l6) (road l10 l17) (road l10 l6) (road l10 l8) (road l11 l24) (road l11 l3) (road l12 l30) (road l12 l7) (road l13 l4) (road l13 l9) (road l14 l26) (road l14 l7) (road l15 l22) (road l15 l23) (road l16 l30) (road l16 l9) (road l17 l10) (road l17 l18) (road l18 l17) (road l18 l29) (road l19 l2) (road l19 l27) (road l2 l19) (road l2 l5) (road l20 l22) (road l20 l28) (road l21 l25) (road l21 l5) (road l22 l15) (road l22 l20) (road l23 l15) (road l23 l4) (road l24 l11) (road l24 l26) (road l25 l0) (road l25 l21) (road l26 l14) (road l26 l24) (road l27 l19) (road l27 l29) (road l28 l0) (road l28 l20) (road l29 l18) (road l29 l27) (road l3 l11) (road l3 l8) (road l30 l12) (road l30 l16) (road l4 l13) (road l4 l23) (road l5 l2) (road l5 l21) (road l6 l1) (road l6 l10) (road l7 l12) (road l7 l14) (road l8 l10) (road l8 l3) (road l9 l13) (road l9 l16) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l2) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l28) (spare-in l3) (spare-in l30) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l8) (spare-in l9) (vehicle-at l18))
    (:goal (vehicle-at l1))
)