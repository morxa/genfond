(define (problem tireworld-023-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l13) (road l0 l19) (road l0 l3) (road l1 l2) (road l1 l9) (road l10 l15) (road l11 l17) (road l11 l21) (road l12 l15) (road l12 l20) (road l12 l5) (road l12 l6) (road l13 l0) (road l13 l17) (road l14 l20) (road l14 l4) (road l14 l6) (road l15 l10) (road l15 l12) (road l16 l8) (road l16 l9) (road l17 l11) (road l17 l13) (road l17 l5) (road l17 l7) (road l18 l19) (road l18 l21) (road l19 l0) (road l19 l18) (road l2 l1) (road l2 l7) (road l20 l12) (road l20 l14) (road l20 l22) (road l21 l11) (road l21 l18) (road l22 l20) (road l22 l3) (road l3 l0) (road l3 l22) (road l4 l14) (road l4 l8) (road l5 l12) (road l5 l17) (road l6 l12) (road l6 l14) (road l7 l17) (road l7 l2) (road l8 l16) (road l8 l4) (road l9 l1) (road l9 l16) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l10))
    (:goal (vehicle-at l22))
)