(define (problem delivery-1x3-1-1-2)
    (:domain delivery)
    (:requirements :typing)
    (:objects c_0_0 c_0_1 c_0_2 - cell p0 - package t0 - truck)
    (:init (adjacent c_0_0 c_0_1) (adjacent c_0_1 c_0_0) (adjacent c_0_1 c_0_2) (adjacent c_0_2 c_0_1) (at p0 c_0_0) (at t0 c_0_0) (empty t0))
    (:goal (and (at p0 c_0_2)))
)