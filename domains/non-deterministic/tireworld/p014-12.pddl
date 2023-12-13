(define (problem tireworld-014-12)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l11) (road l0 l9) (road l1 l0) (road l1 l13) (road l1 l2) (road l1 l7) (road l10 l11) (road l10 l12) (road l10 l3) (road l10 l9) (road l11 l0) (road l11 l10) (road l11 l12) (road l11 l2) (road l12 l10) (road l12 l11) (road l12 l5) (road l13 l1) (road l13 l3) (road l2 l1) (road l2 l11) (road l2 l3) (road l2 l4) (road l3 l10) (road l3 l13) (road l3 l2) (road l3 l4) (road l4 l2) (road l4 l3) (road l4 l8) (road l5 l12) (road l5 l6) (road l5 l9) (road l6 l5) (road l6 l7) (road l6 l8) (road l7 l1) (road l7 l6) (road l7 l8) (road l8 l4) (road l8 l6) (road l8 l7) (road l9 l0) (road l9 l10) (road l9 l5) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l8) (spare-in l9) (vehicle-at l6))
    (:goal (vehicle-at l7))
)