(define (problem tireworld-031-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l18) (road l1 l20) (road l1 l9) (road l10 l11) (road l10 l22) (road l11 l10) (road l11 l27) (road l11 l8) (road l12 l16) (road l12 l5) (road l13 l15) (road l13 l19) (road l14 l8) (road l14 l9) (road l15 l13) (road l15 l30) (road l16 l12) (road l16 l21) (road l17 l27) (road l18 l0) (road l18 l3) (road l18 l6) (road l19 l13) (road l19 l21) (road l2 l4) (road l2 l7) (road l20 l0) (road l20 l1) (road l21 l16) (road l21 l19) (road l22 l10) (road l22 l28) (road l23 l24) (road l23 l4) (road l24 l29) (road l25 l29) (road l25 l30) (road l26 l13) (road l26 l27) (road l27 l11) (road l27 l17) (road l27 l26) (road l28 l22) (road l28 l5) (road l29 l25) (road l3 l18) (road l3 l7) (road l30 l15) (road l30 l25) (road l4 l2) (road l4 l23) (road l5 l12) (road l5 l28) (road l6 l18) (road l6 l7) (road l7 l2) (road l7 l3) (road l7 l6) (road l8 l11) (road l8 l14) (road l9 l1) (road l9 l14) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l18) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l24) (spare-in l25) (spare-in l27) (spare-in l28) (spare-in l3) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l19))
    (:goal (vehicle-at l17))
)