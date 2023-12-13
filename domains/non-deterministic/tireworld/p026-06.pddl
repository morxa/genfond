(define (problem tireworld-026-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l17) (road l0 l25) (road l1 l13) (road l1 l25) (road l10 l18) (road l10 l8) (road l11 l20) (road l11 l9) (road l12 l18) (road l12 l23) (road l12 l4) (road l13 l1) (road l13 l14) (road l14 l13) (road l14 l18) (road l15 l22) (road l15 l24) (road l16 l19) (road l16 l22) (road l17 l0) (road l17 l3) (road l18 l10) (road l18 l12) (road l18 l14) (road l18 l2) (road l19 l16) (road l19 l2) (road l19 l20) (road l2 l18) (road l2 l19) (road l2 l7) (road l20 l11) (road l20 l19) (road l21 l3) (road l21 l5) (road l22 l15) (road l22 l16) (road l22 l7) (road l23 l12) (road l23 l5) (road l24 l15) (road l24 l5) (road l25 l0) (road l25 l1) (road l3 l17) (road l3 l21) (road l4 l12) (road l4 l6) (road l5 l21) (road l5 l23) (road l5 l24) (road l6 l4) (road l6 l9) (road l7 l2) (road l7 l22) (road l8 l10) (road l9 l11) (road l9 l6) (spare-in l10) (spare-in l14) (spare-in l16) (spare-in l18) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l3) (spare-in l7) (spare-in l9) (vehicle-at l12))
    (:goal (vehicle-at l8))
)