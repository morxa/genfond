(define (problem tireworld-022-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l8) (road l1 l19) (road l1 l8) (road l10 l15) (road l10 l8) (road l11 l12) (road l11 l21) (road l12 l11) (road l12 l19) (road l13 l18) (road l13 l20) (road l14 l16) (road l14 l3) (road l15 l10) (road l15 l19) (road l16 l14) (road l16 l2) (road l17 l4) (road l17 l6) (road l18 l13) (road l18 l4) (road l19 l1) (road l19 l12) (road l19 l15) (road l19 l3) (road l2 l16) (road l2 l20) (road l20 l13) (road l20 l2) (road l21 l11) (road l21 l7) (road l3 l14) (road l3 l19) (road l4 l17) (road l4 l18) (road l5 l9) (road l6 l17) (road l6 l9) (road l7 l21) (road l7 l9) (road l8 l0) (road l8 l1) (road l8 l10) (road l9 l5) (road l9 l6) (road l9 l7) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l3) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l0))
    (:goal (vehicle-at l5))
)