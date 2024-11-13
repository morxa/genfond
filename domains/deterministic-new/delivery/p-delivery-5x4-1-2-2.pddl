(define (problem delivery-5x4-1-2-2)
    (:domain delivery)
    (:requirements :typing)
    (:objects c_0_0 c_0_1 c_0_2 c_0_3 c_1_0 c_1_1 c_1_2 c_1_3 c_2_0 c_2_1 c_2_2 c_2_3 c_3_0 c_3_1 c_3_2 c_3_3 c_4_0 c_4_1 c_4_2 c_4_3 - cell p0 p1 - package t0 - truck)
    (:init (adjacent c_0_0 c_0_1) (adjacent c_0_0 c_1_0) (adjacent c_0_1 c_0_0) (adjacent c_0_1 c_0_2) (adjacent c_0_1 c_1_1) (adjacent c_0_2 c_0_1) (adjacent c_0_2 c_0_3) (adjacent c_0_2 c_1_2) (adjacent c_0_3 c_0_2) (adjacent c_0_3 c_1_3) (adjacent c_1_0 c_0_0) (adjacent c_1_0 c_1_1) (adjacent c_1_0 c_2_0) (adjacent c_1_1 c_0_1) (adjacent c_1_1 c_1_0) (adjacent c_1_1 c_1_2) (adjacent c_1_1 c_2_1) (adjacent c_1_2 c_0_2) (adjacent c_1_2 c_1_1) (adjacent c_1_2 c_1_3) (adjacent c_1_2 c_2_2) (adjacent c_1_3 c_0_3) (adjacent c_1_3 c_1_2) (adjacent c_1_3 c_2_3) (adjacent c_2_0 c_1_0) (adjacent c_2_0 c_2_1) (adjacent c_2_0 c_3_0) (adjacent c_2_1 c_1_1) (adjacent c_2_1 c_2_0) (adjacent c_2_1 c_2_2) (adjacent c_2_1 c_3_1) (adjacent c_2_2 c_1_2) (adjacent c_2_2 c_2_1) (adjacent c_2_2 c_2_3) (adjacent c_2_2 c_3_2) (adjacent c_2_3 c_1_3) (adjacent c_2_3 c_2_2) (adjacent c_2_3 c_3_3) (adjacent c_3_0 c_2_0) (adjacent c_3_0 c_3_1) (adjacent c_3_0 c_4_0) (adjacent c_3_1 c_2_1) (adjacent c_3_1 c_3_0) (adjacent c_3_1 c_3_2) (adjacent c_3_1 c_4_1) (adjacent c_3_2 c_2_2) (adjacent c_3_2 c_3_1) (adjacent c_3_2 c_3_3) (adjacent c_3_2 c_4_2) (adjacent c_3_3 c_2_3) (adjacent c_3_3 c_3_2) (adjacent c_3_3 c_4_3) (adjacent c_4_0 c_3_0) (adjacent c_4_0 c_4_1) (adjacent c_4_1 c_3_1) (adjacent c_4_1 c_4_0) (adjacent c_4_1 c_4_2) (adjacent c_4_2 c_3_2) (adjacent c_4_2 c_4_1) (adjacent c_4_2 c_4_3) (adjacent c_4_3 c_3_3) (adjacent c_4_3 c_4_2) (at p0 c_3_1) (at p1 c_1_0) (at t0 c_3_3) (empty t0))
    (:goal (and (at p0 c_2_2) (at p1 c_2_2)))
)