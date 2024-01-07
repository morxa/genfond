(define (problem tireworld-026-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l3) (road l1 l16) (road l1 l18) (road l10 l22) (road l10 l25) (road l11 l4) (road l12 l6) (road l13 l15) (road l14 l21) (road l14 l23) (road l15 l19) (road l16 l3) (road l17 l20) (road l18 l1) (road l18 l25) (road l19 l1) (road l19 l15) (road l2 l3) (road l20 l12) (road l21 l14) (road l21 l7) (road l21 l9) (road l22 l24) (road l23 l14) (road l24 l22) (road l25 l10) (road l3 l24) (road l3 l5) (road l4 l8) (road l5 l23) (road l6 l0) (road l6 l12) (road l7 l13) (road l8 l2) (road l9 l11) (spare-in l0) (spare-in l12) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l24) (spare-in l25) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l9) (vehicle-at l17))
    (:goal (vehicle-at l22))
)