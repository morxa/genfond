(define (problem tireworld-024-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l0 l19) (road l0 l5) (road l1 l2) (road l1 l9) (road l10 l14) (road l10 l18) (road l11 l13) (road l11 l9) (road l12 l0) (road l12 l18) (road l12 l22) (road l13 l11) (road l13 l6) (road l14 l10) (road l14 l19) (road l15 l20) (road l15 l6) (road l16 l22) (road l16 l4) (road l17 l19) (road l17 l2) (road l18 l10) (road l18 l12) (road l18 l23) (road l18 l7) (road l19 l0) (road l19 l14) (road l19 l17) (road l2 l1) (road l2 l17) (road l20 l15) (road l20 l21) (road l21 l20) (road l21 l8) (road l22 l12) (road l22 l16) (road l23 l18) (road l23 l3) (road l3 l23) (road l3 l7) (road l4 l16) (road l4 l8) (road l5 l0) (road l6 l13) (road l6 l15) (road l7 l18) (road l7 l3) (road l8 l21) (road l8 l4) (road l9 l1) (road l9 l11) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l6) (spare-in l7) (vehicle-at l5))
    (:goal (vehicle-at l3))
)