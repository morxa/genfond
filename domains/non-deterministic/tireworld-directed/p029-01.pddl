(define (problem tireworld-029-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l16) (road l0 l5) (road l1 l26) (road l1 l28) (road l10 l14) (road l10 l8) (road l11 l19) (road l12 l11) (road l12 l26) (road l12 l4) (road l13 l16) (road l14 l10) (road l14 l17) (road l15 l8) (road l16 l0) (road l16 l13) (road l17 l14) (road l18 l20) (road l18 l27) (road l19 l20) (road l2 l21) (road l20 l18) (road l20 l19) (road l21 l2) (road l21 l24) (road l22 l4) (road l22 l9) (road l23 l13) (road l23 l3) (road l24 l21) (road l24 l9) (road l25 l17) (road l25 l27) (road l26 l1) (road l27 l25) (road l28 l1) (road l28 l7) (road l3 l23) (road l3 l6) (road l4 l12) (road l4 l22) (road l5 l0) (road l5 l15) (road l6 l3) (road l7 l18) (road l8 l10) (road l9 l22) (road l9 l6) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l3) (spare-in l5) (spare-in l6) (spare-in l8) (spare-in l9) (vehicle-at l2))
    (:goal (vehicle-at l10))
)