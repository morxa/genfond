(define (problem tireworld-019-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l4) (road l1 l8) (road l10 l6) (road l11 l16) (road l12 l5) (road l13 l6) (road l14 l12) (road l14 l5) (road l14 l9) (road l15 l3) (road l16 l7) (road l17 l2) (road l18 l11) (road l2 l13) (road l2 l17) (road l3 l14) (road l4 l15) (road l5 l0) (road l5 l18) (road l6 l10) (road l7 l17) (road l8 l4) (road l9 l1) (spare-in l11) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l1))
    (:goal (vehicle-at l10))
)