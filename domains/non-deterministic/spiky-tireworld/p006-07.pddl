(define (problem spiky-tireworld-006-07)
    (:domain sptire)
    (:requirements :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 - location t0 - tire)
    (:init (not-flattire) (not-hasspare) (road l0 l1) (road l1 l0) (road l1 l2) (road l2 l1) (road l4 l5) (road l5 l4) (spiky-road l0 l5) (spiky-road l2 l3) (spiky-road l3 l2) (spiky-road l3 l4) (spiky-road l4 l3) (spiky-road l5 l0) (tire-at t0 l1) (vehicle-at l0))
    (:goal (vehicle-at l5))
)