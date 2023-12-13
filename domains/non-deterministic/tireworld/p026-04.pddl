(define (problem tireworld-026-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l20) (road l0 l23) (road l1 l18) (road l1 l24) (road l10 l22) (road l10 l3) (road l11 l16) (road l11 l2) (road l12 l4) (road l12 l9) (road l13 l18) (road l13 l21) (road l14 l2) (road l14 l7) (road l15 l6) (road l15 l9) (road l16 l11) (road l16 l8) (road l17 l6) (road l18 l1) (road l18 l13) (road l19 l20) (road l19 l7) (road l2 l11) (road l2 l14) (road l2 l6) (road l20 l0) (road l20 l19) (road l20 l5) (road l21 l13) (road l21 l25) (road l22 l10) (road l22 l8) (road l23 l0) (road l23 l6) (road l24 l1) (road l24 l4) (road l25 l21) (road l25 l3) (road l3 l10) (road l3 l25) (road l3 l5) (road l4 l12) (road l4 l24) (road l5 l20) (road l5 l3) (road l6 l15) (road l6 l17) (road l6 l2) (road l6 l23) (road l7 l14) (road l7 l19) (road l8 l16) (road l8 l22) (road l9 l12) (road l9 l15) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l8) (vehicle-at l2))
    (:goal (vehicle-at l17))
)