(define (problem tireworld-014-7)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l12) (road l1 l0) (road l1 l2) (road l11 l12) (road l12 l0) (road l12 l11) (road l12 l13) (road l13 l12) (road l2 l1) (road l2 l3) (road l3 l2) (road l6 l7) (road l7 l6) (road l7 l8) (road l8 l7) (road l8 l9) (road l9 l8) (vehicle-at l0))
    (:goal (vehicle-at l12))
)