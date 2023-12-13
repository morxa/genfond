(define (problem tireworld-022-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l13) (road l0 l19) (road l0 l7) (road l1 l11) (road l1 l16) (road l1 l17) (road l10 l14) (road l10 l18) (road l11 l1) (road l11 l14) (road l11 l8) (road l12 l18) (road l12 l9) (road l13 l0) (road l13 l21) (road l14 l10) (road l14 l11) (road l15 l21) (road l15 l9) (road l16 l1) (road l17 l1) (road l17 l5) (road l18 l10) (road l18 l12) (road l19 l0) (road l19 l2) (road l19 l20) (road l2 l19) (road l2 l5) (road l20 l19) (road l20 l7) (road l21 l13) (road l21 l15) (road l3 l6) (road l3 l9) (road l4 l5) (road l4 l8) (road l5 l17) (road l5 l2) (road l5 l4) (road l6 l3) (road l7 l0) (road l7 l20) (road l8 l11) (road l8 l4) (road l9 l12) (road l9 l15) (road l9 l3) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l18) (spare-in l21) (spare-in l3) (spare-in l4) (spare-in l8) (spare-in l9) (vehicle-at l6))
    (:goal (vehicle-at l16))
)