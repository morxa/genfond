(define (problem delivery-1x2-1-2-1)
    (:domain delivery)
    (:requirements :typing)
    (:objects c_0_0 c_0_1 - cell p0 p1 - package t0 - truck)
    (:init (adjacent c_0_0 c_0_1) (adjacent c_0_1 c_0_0) (at p0 c_0_1) (at p1 c_0_0) (at t0 c_0_0) (empty t0))
    (:goal (and (at p0 c_0_1) (at p1 c_0_1)))
)