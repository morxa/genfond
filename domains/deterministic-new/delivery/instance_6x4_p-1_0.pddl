
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Instance file automatically generated by the Tarski FSTRIPS writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem delivery_6x4_p-1)
    (:domain delivery)

    (:objects
        c_0_0 c_0_1 c_0_2 c_0_3 c_1_0 c_1_1 c_1_2 c_1_3 c_2_0 c_2_1 c_2_2 c_2_3 c_3_0 c_3_1 c_3_2 c_3_3 c_4_0 c_4_1 c_4_2 c_4_3 c_5_0 c_5_1 c_5_2 c_5_3 - cell
        p1 - package
        t1 - truck
    )

    (:init
        (adjacent c_4_1 c_4_2)
        (adjacent c_2_0 c_1_0)
        (adjacent c_4_3 c_5_3)
        (adjacent c_2_3 c_2_2)
        (adjacent c_2_3 c_3_3)
        (adjacent c_0_1 c_1_1)
        (adjacent c_1_1 c_0_1)
        (adjacent c_5_3 c_5_2)
        (adjacent c_5_0 c_5_1)
        (adjacent c_3_3 c_2_3)
        (adjacent c_0_0 c_0_1)
        (adjacent c_3_0 c_4_0)
        (adjacent c_3_1 c_2_1)
        (adjacent c_3_1 c_3_2)
        (adjacent c_3_1 c_4_1)
        (adjacent c_5_2 c_5_1)
        (adjacent c_1_3 c_0_3)
        (adjacent c_2_2 c_2_3)
        (adjacent c_4_1 c_5_1)
        (adjacent c_0_1 c_0_2)
        (adjacent c_1_2 c_2_2)
        (adjacent c_1_2 c_1_3)
        (adjacent c_2_0 c_2_1)
        (adjacent c_3_3 c_4_3)
        (adjacent c_3_0 c_3_1)
        (adjacent c_3_3 c_3_2)
        (adjacent c_1_2 c_1_1)
        (adjacent c_0_3 c_1_3)
        (adjacent c_0_2 c_1_2)
        (adjacent c_1_3 c_2_3)
        (adjacent c_2_1 c_3_1)
        (adjacent c_3_2 c_3_1)
        (adjacent c_1_0 c_0_0)
        (adjacent c_4_3 c_4_2)
        (adjacent c_3_2 c_4_2)
        (adjacent c_5_2 c_5_3)
        (adjacent c_5_1 c_5_2)
        (adjacent c_5_1 c_4_1)
        (adjacent c_0_2 c_0_1)
        (adjacent c_4_0 c_3_0)
        (adjacent c_5_1 c_5_0)
        (adjacent c_1_0 c_1_1)
        (adjacent c_1_0 c_2_0)
        (adjacent c_2_2 c_2_1)
        (adjacent c_2_2 c_3_2)
        (adjacent c_2_2 c_1_2)
        (adjacent c_5_0 c_4_0)
        (adjacent c_1_2 c_0_2)
        (adjacent c_1_1 c_1_0)
        (adjacent c_2_1 c_2_2)
        (adjacent c_0_0 c_1_0)
        (adjacent c_4_3 c_3_3)
        (adjacent c_3_2 c_2_2)
        (adjacent c_3_2 c_3_3)
        (adjacent c_0_2 c_0_3)
        (adjacent c_3_1 c_3_0)
        (adjacent c_4_2 c_3_2)
        (adjacent c_4_2 c_4_1)
        (adjacent c_5_3 c_4_3)
        (adjacent c_2_1 c_1_1)
        (adjacent c_3_0 c_2_0)
        (adjacent c_1_3 c_1_2)
        (adjacent c_4_1 c_4_0)
        (adjacent c_0_3 c_0_2)
        (adjacent c_4_2 c_4_3)
        (adjacent c_4_2 c_5_2)
        (adjacent c_2_1 c_2_0)
        (adjacent c_2_0 c_3_0)
        (adjacent c_4_0 c_4_1)
        (adjacent c_0_1 c_0_0)
        (adjacent c_1_1 c_2_1)
        (adjacent c_5_2 c_4_2)
        (adjacent c_4_1 c_3_1)
        (adjacent c_4_0 c_5_0)
        (adjacent c_2_3 c_1_3)
        (adjacent c_1_1 c_1_2)
        (at t1 c_0_0)
        (at p1 c_5_1)
        (empty t1)
    )

    (:goal
        (at p1 c_1_3)
    )

    
    
    
)

