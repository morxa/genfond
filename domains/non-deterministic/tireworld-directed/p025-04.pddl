(define (problem tireworld-025-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l1 l21) (road l10 l16) (road l11 l3) (road l12 l0) (road l12 l19) (road l12 l24) (road l13 l9) (road l14 l12) (road l14 l15) (road l15 l14) (road l16 l22) (road l16 l5) (road l17 l4) (road l18 l1) (road l18 l11) (road l19 l8) (road l2 l10) (road l2 l3) (road l20 l22) (road l21 l12) (road l22 l16) (road l23 l13) (road l24 l8) (road l3 l2) (road l4 l7) (road l5 l15) (road l5 l16) (road l6 l17) (road l6 l20) (road l7 l18) (road l8 l0) (road l8 l23) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l21) (spare-in l23) (spare-in l4) (spare-in l7) (spare-in l8) (vehicle-at l6))
    (:goal (vehicle-at l9))
)