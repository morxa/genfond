(define (problem delivery-3x4-1-3-1)
    (:domain delivery)
    (:requirements :typing)
    (:objects c_0_0 c_0_1 c_0_2 c_0_3 c_1_0 c_1_1 c_1_2 c_1_3 c_2_0 c_2_1 c_2_2 c_2_3 - cell p0 p1 p2 - package t0 - truck)
    (:init (adjacent c_0_0 c_0_1) (adjacent c_0_0 c_1_0) (adjacent c_0_1 c_0_0) (adjacent c_0_1 c_0_2) (adjacent c_0_1 c_1_1) (adjacent c_0_2 c_0_1) (adjacent c_0_2 c_0_3) (adjacent c_0_2 c_1_2) (adjacent c_0_3 c_0_2) (adjacent c_0_3 c_1_3) (adjacent c_1_0 c_0_0) (adjacent c_1_0 c_1_1) (adjacent c_1_0 c_2_0) (adjacent c_1_1 c_0_1) (adjacent c_1_1 c_1_0) (adjacent c_1_1 c_1_2) (adjacent c_1_1 c_2_1) (adjacent c_1_2 c_0_2) (adjacent c_1_2 c_1_1) (adjacent c_1_2 c_1_3) (adjacent c_1_2 c_2_2) (adjacent c_1_3 c_0_3) (adjacent c_1_3 c_1_2) (adjacent c_1_3 c_2_3) (adjacent c_2_0 c_1_0) (adjacent c_2_0 c_2_1) (adjacent c_2_1 c_1_1) (adjacent c_2_1 c_2_0) (adjacent c_2_1 c_2_2) (adjacent c_2_2 c_1_2) (adjacent c_2_2 c_2_1) (adjacent c_2_2 c_2_3) (adjacent c_2_3 c_1_3) (adjacent c_2_3 c_2_2) (at p0 c_2_1) (at p1 c_1_0) (at p2 c_2_1) (at t0 c_0_0) (empty t0))
    (:goal (and (at p0 c_2_2) (at p1 c_2_2) (at p2 c_2_2)))
)