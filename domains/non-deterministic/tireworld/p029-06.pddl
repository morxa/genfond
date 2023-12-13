(define (problem tireworld-029-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l13) (road l1 l0) (road l1 l20) (road l10 l24) (road l10 l7) (road l11 l16) (road l11 l19) (road l12 l15) (road l12 l24) (road l13 l0) (road l13 l18) (road l14 l2) (road l14 l3) (road l15 l12) (road l15 l28) (road l16 l11) (road l16 l2) (road l17 l19) (road l17 l5) (road l18 l13) (road l18 l28) (road l19 l11) (road l19 l17) (road l19 l21) (road l2 l14) (road l2 l16) (road l20 l1) (road l20 l21) (road l21 l19) (road l21 l20) (road l22 l23) (road l22 l4) (road l23 l22) (road l23 l27) (road l24 l10) (road l24 l12) (road l25 l5) (road l25 l6) (road l26 l28) (road l26 l9) (road l27 l23) (road l27 l8) (road l28 l15) (road l28 l18) (road l28 l26) (road l3 l14) (road l3 l7) (road l4 l22) (road l4 l5) (road l5 l17) (road l5 l25) (road l5 l4) (road l6 l25) (road l6 l8) (road l6 l9) (road l7 l10) (road l7 l3) (road l8 l27) (road l8 l6) (road l9 l26) (road l9 l6) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l24) (spare-in l26) (spare-in l28) (spare-in l3) (spare-in l4) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l5))
    (:goal (vehicle-at l17))
)