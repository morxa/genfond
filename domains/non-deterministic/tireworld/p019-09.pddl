(define (problem tireworld-019-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l3) (road l0 l6) (road l1 l17) (road l1 l4) (road l10 l3) (road l11 l12) (road l11 l2) (road l12 l11) (road l12 l13) (road l13 l12) (road l13 l17) (road l14 l8) (road l14 l9) (road l15 l18) (road l15 l7) (road l16 l5) (road l16 l7) (road l17 l1) (road l17 l13) (road l18 l15) (road l18 l4) (road l18 l6) (road l2 l11) (road l2 l9) (road l3 l0) (road l3 l10) (road l4 l1) (road l4 l18) (road l5 l16) (road l5 l6) (road l6 l0) (road l6 l18) (road l6 l5) (road l7 l15) (road l7 l16) (road l8 l14) (road l9 l14) (road l9 l2) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l9) (vehicle-at l10))
    (:goal (vehicle-at l8))
)