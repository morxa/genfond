(define (problem tireworld-017-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l2 l3 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l16) (road l0 l4) (road l1 l5) (road l1 l8) (road l10 l11) (road l11 l10) (road l11 l12) (road l11 l6) (road l12 l11) (road l12 l13) (road l12 l16) (road l13 l12) (road l13 l14) (road l13 l3) (road l13 l9) (road l14 l13) (road l14 l15) (road l15 l14) (road l15 l16) (road l15 l2) (road l15 l5) (road l16 l0) (road l16 l12) (road l16 l15) (road l2 l15) (road l2 l3) (road l2 l4) (road l3 l13) (road l3 l2) (road l3 l4) (road l3 l6) (road l4 l0) (road l4 l2) (road l4 l3) (road l4 l5) (road l5 l1) (road l5 l15) (road l5 l4) (road l5 l6) (road l6 l11) (road l6 l3) (road l6 l5) (road l6 l7) (road l7 l6) (road l7 l8) (road l7 l9) (road l8 l1) (road l8 l7) (road l8 l9) (road l9 l13) (road l9 l7) (road l9 l8) (spare-in l0) (spare-in l1) (spare-in l13) (spare-in l15) (spare-in l16) (spare-in l2) (spare-in l3) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l12))
    (:goal (vehicle-at l11))
)