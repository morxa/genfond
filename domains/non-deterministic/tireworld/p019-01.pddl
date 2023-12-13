(define (problem tireworld-019-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l7) (road l0 l9) (road l1 l13) (road l1 l5) (road l10 l14) (road l10 l6) (road l11 l17) (road l11 l9) (road l12 l16) (road l13 l1) (road l14 l10) (road l14 l15) (road l15 l14) (road l15 l4) (road l16 l12) (road l16 l17) (road l16 l2) (road l17 l11) (road l17 l16) (road l17 l9) (road l18 l3) (road l18 l7) (road l2 l16) (road l2 l3) (road l3 l18) (road l3 l2) (road l4 l15) (road l4 l8) (road l5 l1) (road l5 l8) (road l6 l10) (road l6 l9) (road l7 l0) (road l7 l18) (road l8 l4) (road l8 l5) (road l9 l0) (road l9 l11) (road l9 l17) (road l9 l6) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l13))
    (:goal (vehicle-at l12))
)