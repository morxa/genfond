(define (problem delivery-2x2-1-1-0)
    (:domain delivery)
    (:requirements :typing)
    (:objects c_0_0 c_0_1 c_1_0 c_1_1 - cell p0 - package t0 - truck)
    (:init (adjacent c_0_0 c_0_1) (adjacent c_0_0 c_1_0) (adjacent c_0_1 c_0_0) (adjacent c_0_1 c_1_1) (adjacent c_1_0 c_0_0) (adjacent c_1_0 c_1_1) (adjacent c_1_1 c_0_1) (adjacent c_1_1 c_1_0) (at p0 c_0_0) (at t0 c_1_1) (empty t0))
    (:goal (and (at p0 c_0_1)))
)