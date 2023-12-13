(define (problem tireworld-019-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l10) (road l1 l15) (road l1 l5) (road l10 l0) (road l10 l14) (road l11 l17) (road l11 l8) (road l12 l5) (road l12 l7) (road l13 l18) (road l14 l10) (road l14 l7) (road l15 l1) (road l15 l18) (road l16 l6) (road l16 l7) (road l17 l11) (road l17 l2) (road l18 l13) (road l18 l15) (road l18 l4) (road l18 l6) (road l2 l17) (road l2 l3) (road l3 l2) (road l3 l9) (road l4 l18) (road l4 l8) (road l5 l1) (road l5 l12) (road l5 l9) (road l6 l16) (road l6 l18) (road l7 l12) (road l7 l14) (road l7 l16) (road l8 l11) (road l8 l4) (road l9 l3) (road l9 l5) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l18) (spare-in l4) (spare-in l5) (spare-in l7) (vehicle-at l13))
    (:goal (vehicle-at l0))
)