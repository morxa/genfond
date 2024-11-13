(define (problem delivery-4x2-1-3-2)
    (:domain delivery)
    (:requirements :typing)
    (:objects c_0_0 c_0_1 c_1_0 c_1_1 c_2_0 c_2_1 c_3_0 c_3_1 - cell p0 p1 p2 - package t0 - truck)
    (:init (adjacent c_0_0 c_0_1) (adjacent c_0_0 c_1_0) (adjacent c_0_1 c_0_0) (adjacent c_0_1 c_1_1) (adjacent c_1_0 c_0_0) (adjacent c_1_0 c_1_1) (adjacent c_1_0 c_2_0) (adjacent c_1_1 c_0_1) (adjacent c_1_1 c_1_0) (adjacent c_1_1 c_2_1) (adjacent c_2_0 c_1_0) (adjacent c_2_0 c_2_1) (adjacent c_2_0 c_3_0) (adjacent c_2_1 c_1_1) (adjacent c_2_1 c_2_0) (adjacent c_2_1 c_3_1) (adjacent c_3_0 c_2_0) (adjacent c_3_0 c_3_1) (adjacent c_3_1 c_2_1) (adjacent c_3_1 c_3_0) (at p0 c_0_1) (at p1 c_1_1) (at p2 c_1_1) (at t0 c_3_1) (empty t0))
    (:goal (and (at p0 c_0_1) (at p1 c_0_1) (at p2 c_0_1)))
)