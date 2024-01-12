(define (problem spiky-tireworld-006-03)
    (:domain sptire)
    (:requirements :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 - location t0 t1 t2 - tire)
    (:init (not-flattire) (not-hasspare) (road l0 l1) (road l1 l0) (road l1 l2) (road l2 l1) (road l2 l3) (road l3 l2) (spiky-road l0 l5) (spiky-road l3 l4) (spiky-road l4 l3) (spiky-road l4 l5) (spiky-road l5 l0) (spiky-road l5 l4) (tire-at t0 l1) (tire-at t1 l1) (tire-at t2 l1) (vehicle-at l0))
    (:goal (vehicle-at l5))
)