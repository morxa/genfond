(define (problem tireworld-020-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l8) (road l1 l16) (road l10 l12) (road l10 l15) (road l10 l17) (road l11 l12) (road l11 l19) (road l12 l11) (road l13 l19) (road l14 l13) (road l15 l10) (road l15 l5) (road l16 l3) (road l17 l0) (road l18 l4) (road l18 l6) (road l19 l11) (road l2 l7) (road l3 l9) (road l4 l14) (road l5 l18) (road l6 l2) (road l7 l1) (road l8 l11) (road l9 l12) (road l9 l3) (vehicle-at l10))
    (:goal (vehicle-at l12))
)