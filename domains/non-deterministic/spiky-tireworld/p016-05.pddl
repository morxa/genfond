(define (problem spiky-tireworld-016-05)
    (:domain sptire)
    (:requirements :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l2 l3 l4 l5 l6 l7 l8 l9 - location t0 - tire)
    (:init (not-flattire) (not-hasspare) (road l0 l1) (road l0 l13) (road l1 l0) (road l10 l11) (road l10 l9) (road l11 l10) (road l11 l12) (road l12 l11) (road l12 l15) (road l13 l0) (road l13 l14) (road l14 l13) (road l15 l12) (road l3 l4) (road l4 l3) (road l4 l5) (road l5 l4) (road l5 l6) (road l6 l5) (road l6 l7) (road l7 l6) (road l7 l8) (road l8 l7) (road l8 l9) (road l9 l10) (road l9 l8) (spiky-road l1 l2) (spiky-road l14 l15) (spiky-road l15 l14) (spiky-road l2 l1) (spiky-road l2 l3) (spiky-road l3 l2) (tire-at t0 l1) (vehicle-at l0))
    (:goal (vehicle-at l15))
)