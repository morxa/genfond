(define (problem tireworld-018-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l17) (road l0 l6) (road l0 l7) (road l1 l0) (road l1 l16) (road l10 l13) (road l10 l4) (road l11 l15) (road l11 l4) (road l12 l15) (road l12 l3) (road l13 l10) (road l13 l17) (road l14 l16) (road l14 l5) (road l15 l11) (road l15 l12) (road l16 l1) (road l16 l14) (road l17 l0) (road l17 l13) (road l2 l9) (road l3 l12) (road l3 l6) (road l3 l8) (road l4 l10) (road l4 l11) (road l5 l14) (road l5 l9) (road l6 l0) (road l6 l3) (road l7 l0) (road l7 l9) (road l8 l3) (road l8 l9) (road l9 l2) (road l9 l5) (road l9 l7) (road l9 l8) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l10))
    (:goal (vehicle-at l2))
)