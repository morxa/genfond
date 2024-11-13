(define (problem delivery-1x4-1-3-0)
    (:domain delivery)
    (:requirements :typing)
    (:objects c_0_0 c_0_1 c_0_2 c_0_3 - cell p0 p1 p2 - package t0 - truck)
    (:init (adjacent c_0_0 c_0_1) (adjacent c_0_1 c_0_0) (adjacent c_0_1 c_0_2) (adjacent c_0_2 c_0_1) (adjacent c_0_2 c_0_3) (adjacent c_0_3 c_0_2) (at p0 c_0_3) (at p1 c_0_0) (at p2 c_0_0) (at t0 c_0_2) (empty t0))
    (:goal (and (at p0 c_0_0) (at p1 c_0_0) (at p2 c_0_0)))
)