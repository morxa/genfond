(define (problem delivery-2x5-1-1-2)
    (:domain delivery)
    (:requirements :typing)
    (:objects c_0_0 c_0_1 c_0_2 c_0_3 c_0_4 c_1_0 c_1_1 c_1_2 c_1_3 c_1_4 - cell p0 - package t0 - truck)
    (:init (adjacent c_0_0 c_0_1) (adjacent c_0_0 c_1_0) (adjacent c_0_1 c_0_0) (adjacent c_0_1 c_0_2) (adjacent c_0_1 c_1_1) (adjacent c_0_2 c_0_1) (adjacent c_0_2 c_0_3) (adjacent c_0_2 c_1_2) (adjacent c_0_3 c_0_2) (adjacent c_0_3 c_0_4) (adjacent c_0_3 c_1_3) (adjacent c_0_4 c_0_3) (adjacent c_0_4 c_1_4) (adjacent c_1_0 c_0_0) (adjacent c_1_0 c_1_1) (adjacent c_1_1 c_0_1) (adjacent c_1_1 c_1_0) (adjacent c_1_1 c_1_2) (adjacent c_1_2 c_0_2) (adjacent c_1_2 c_1_1) (adjacent c_1_2 c_1_3) (adjacent c_1_3 c_0_3) (adjacent c_1_3 c_1_2) (adjacent c_1_3 c_1_4) (adjacent c_1_4 c_0_4) (adjacent c_1_4 c_1_3) (at p0 c_0_1) (at t0 c_0_3) (empty t0))
    (:goal (and (at p0 c_0_4)))
)