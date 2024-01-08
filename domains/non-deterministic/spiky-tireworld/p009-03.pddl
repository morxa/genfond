(define (problem tireworld-009-03)
    (:domain sptire)
    (:requirements :non-deterministic :strips :typing)
    (:objects l0 l1 l2 l3 l4 l5 l6 l7 l8 - location t0 t1 t2 - tire)
    (:init (not-flattire) (not-hasspare) (road l0 l1) (road l0 l3) (road l1 l0) (road l3 l0) (road l3 l4) (road l4 l3) (road l4 l5) (road l5 l4) (road l5 l6) (road l6 l5) (road l7 l8) (road l8 l7) (spiky-road l1 l2) (spiky-road l2 l1) (spiky-road l2 l8) (spiky-road l6 l7) (spiky-road l7 l6) (spiky-road l8 l2) (tire-at t0 l1) (tire-at t1 l1) (tire-at t2 l1) (vehicle-at l0))
    (:goal (vehicle-at l8))
)