
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Instance file automatically generated by the Tarski FSTRIPS writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem delivery_8x7_p-1)
    (:domain delivery)

    (:objects
        c_0_0 c_0_1 c_0_2 c_0_3 c_0_4 c_0_5 c_0_6 c_1_0 c_1_1 c_1_2 c_1_3 c_1_4 c_1_5 c_1_6 c_2_0 c_2_1 c_2_2 c_2_3 c_2_4 c_2_5 c_2_6 c_3_0 c_3_1 c_3_2 c_3_3 c_3_4 c_3_5 c_3_6 c_4_0 c_4_1 c_4_2 c_4_3 c_4_4 c_4_5 c_4_6 c_5_0 c_5_1 c_5_2 c_5_3 c_5_4 c_5_5 c_5_6 c_6_0 c_6_1 c_6_2 c_6_3 c_6_4 c_6_5 c_6_6 c_7_0 c_7_1 c_7_2 c_7_3 c_7_4 c_7_5 c_7_6 - cell
        p1 - package
        t1 - truck
    )

    (:init
        (adjacent c_6_0 c_5_0)
        (adjacent c_4_3 c_5_3)
        (adjacent c_1_6 c_1_5)
        (adjacent c_5_0 c_5_1)
        (adjacent c_2_4 c_3_4)
        (adjacent c_7_2 c_7_3)
        (adjacent c_0_0 c_0_1)
        (adjacent c_3_0 c_4_0)
        (adjacent c_5_0 c_6_0)
        (adjacent c_5_2 c_5_1)
        (adjacent c_6_3 c_5_3)
        (adjacent c_2_2 c_2_3)
        (adjacent c_7_1 c_7_0)
        (adjacent c_1_2 c_2_2)
        (adjacent c_7_5 c_7_6)
        (adjacent c_1_2 c_1_3)
        (adjacent c_7_5 c_7_4)
        (adjacent c_7_2 c_7_1)
        (adjacent c_0_4 c_1_4)
        (adjacent c_2_3 c_2_4)
        (adjacent c_1_2 c_1_1)
        (adjacent c_7_0 c_7_1)
        (adjacent c_5_5 c_4_5)
        (adjacent c_7_1 c_6_1)
        (adjacent c_6_4 c_5_4)
        (adjacent c_1_0 c_1_1)
        (adjacent c_7_4 c_6_4)
        (adjacent c_7_5 c_6_5)
        (adjacent c_1_4 c_1_5)
        (adjacent c_3_5 c_3_4)
        (adjacent c_1_2 c_0_2)
        (adjacent c_2_1 c_2_2)
        (adjacent c_0_0 c_1_0)
        (adjacent c_3_2 c_2_2)
        (adjacent c_3_1 c_3_0)
        (adjacent c_5_4 c_6_4)
        (adjacent c_5_3 c_5_2)
        (adjacent c_2_1 c_1_1)
        (adjacent c_3_0 c_2_0)
        (adjacent c_1_3 c_1_2)
        (adjacent c_4_1 c_4_0)
        (adjacent c_0_3 c_0_2)
        (adjacent c_5_3 c_5_4)
        (adjacent c_5_5 c_5_4)
        (adjacent c_2_1 c_2_0)
        (adjacent c_2_0 c_3_0)
        (adjacent c_4_4 c_4_3)
        (adjacent c_1_6 c_2_6)
        (adjacent c_1_1 c_2_1)
        (adjacent c_5_2 c_4_2)
        (adjacent c_2_3 c_2_2)
        (adjacent c_1_1 c_0_1)
        (adjacent c_2_6 c_1_6)
        (adjacent c_6_3 c_6_2)
        (adjacent c_3_1 c_3_2)
        (adjacent c_6_4 c_6_3)
        (adjacent c_0_6 c_0_5)
        (adjacent c_6_5 c_6_6)
        (adjacent c_4_1 c_5_1)
        (adjacent c_3_4 c_3_3)
        (adjacent c_6_4 c_7_4)
        (adjacent c_4_4 c_3_4)
        (adjacent c_6_3 c_7_3)
        (adjacent c_1_3 c_2_3)
        (adjacent c_1_0 c_0_0)
        (adjacent c_2_4 c_2_5)
        (adjacent c_3_2 c_4_2)
        (adjacent c_4_6 c_5_6)
        (adjacent c_5_3 c_6_3)
        (adjacent c_6_1 c_6_2)
        (adjacent c_5_1 c_4_1)
        (adjacent c_0_2 c_0_1)
        (adjacent c_4_0 c_3_0)
        (adjacent c_5_4 c_5_5)
        (adjacent c_5_1 c_5_0)
        (adjacent c_6_5 c_6_4)
        (adjacent c_7_3 c_7_2)
        (adjacent c_2_2 c_3_2)
        (adjacent c_2_5 c_1_5)
        (adjacent c_1_5 c_1_6)
        (adjacent c_6_4 c_6_5)
        (adjacent c_7_4 c_7_5)
        (adjacent c_7_1 c_7_2)
        (adjacent c_6_6 c_5_6)
        (adjacent c_2_6 c_3_6)
        (adjacent c_5_3 c_4_3)
        (adjacent c_1_4 c_0_4)
        (adjacent c_6_0 c_7_0)
        (adjacent c_2_6 c_2_5)
        (adjacent c_4_2 c_5_2)
        (adjacent c_1_4 c_1_3)
        (adjacent c_3_4 c_4_4)
        (adjacent c_1_3 c_1_4)
        (adjacent c_4_0 c_4_1)
        (adjacent c_3_5 c_3_6)
        (adjacent c_7_6 c_7_5)
        (adjacent c_4_0 c_5_0)
        (adjacent c_6_2 c_5_2)
        (adjacent c_5_5 c_6_5)
        (adjacent c_5_4 c_5_3)
        (adjacent c_3_3 c_2_3)
        (adjacent c_7_0 c_6_0)
        (adjacent c_4_1 c_4_2)
        (adjacent c_6_1 c_7_1)
        (adjacent c_3_6 c_2_6)
        (adjacent c_6_0 c_6_1)
        (adjacent c_7_3 c_6_3)
        (adjacent c_3_1 c_2_1)
        (adjacent c_3_1 c_4_1)
        (adjacent c_5_5 c_5_6)
        (adjacent c_6_6 c_7_6)
        (adjacent c_5_4 c_4_4)
        (adjacent c_1_3 c_0_3)
        (adjacent c_5_6 c_6_6)
        (adjacent c_1_4 c_2_4)
        (adjacent c_0_1 c_0_2)
        (adjacent c_7_3 c_7_4)
        (adjacent c_2_0 c_2_1)
        (adjacent c_3_3 c_4_3)
        (adjacent c_0_5 c_0_6)
        (adjacent c_1_5 c_2_5)
        (adjacent c_6_5 c_5_5)
        (adjacent c_6_1 c_6_0)
        (adjacent c_0_2 c_1_2)
        (adjacent c_3_2 c_3_1)
        (adjacent c_4_3 c_4_2)
        (adjacent c_5_2 c_5_3)
        (adjacent c_5_1 c_5_2)
        (adjacent c_6_6 c_6_5)
        (adjacent c_2_2 c_2_1)
        (adjacent c_6_5 c_7_5)
        (adjacent c_6_2 c_6_3)
        (adjacent c_1_1 c_1_0)
        (adjacent c_4_5 c_3_5)
        (adjacent c_3_6 c_3_5)
        (adjacent c_3_2 c_3_3)
        (adjacent c_0_2 c_0_3)
        (adjacent c_4_2 c_3_2)
        (adjacent c_3_3 c_3_4)
        (adjacent c_4_2 c_4_3)
        (adjacent c_4_6 c_3_6)
        (adjacent c_4_1 c_3_1)
        (adjacent c_0_4 c_0_5)
        (adjacent c_7_4 c_7_3)
        (adjacent c_1_1 c_1_2)
        (adjacent c_6_3 c_6_4)
        (adjacent c_2_0 c_1_0)
        (adjacent c_2_3 c_3_3)
        (adjacent c_3_5 c_2_5)
        (adjacent c_0_1 c_1_1)
        (adjacent c_6_2 c_6_1)
        (adjacent c_2_5 c_3_5)
        (adjacent c_6_2 c_7_2)
        (adjacent c_4_5 c_5_5)
        (adjacent c_4_3 c_4_4)
        (adjacent c_2_5 c_2_4)
        (adjacent c_1_5 c_0_5)
        (adjacent c_2_4 c_2_3)
        (adjacent c_5_2 c_6_2)
        (adjacent c_5_6 c_5_5)
        (adjacent c_0_3 c_0_4)
        (adjacent c_3_0 c_3_1)
        (adjacent c_3_3 c_3_2)
        (adjacent c_6_1 c_5_1)
        (adjacent c_0_3 c_1_3)
        (adjacent c_2_1 c_3_1)
        (adjacent c_7_6 c_6_6)
        (adjacent c_2_5 c_2_6)
        (adjacent c_4_5 c_4_6)
        (adjacent c_3_6 c_4_6)
        (adjacent c_0_6 c_1_6)
        (adjacent c_1_0 c_2_0)
        (adjacent c_1_5 c_1_4)
        (adjacent c_2_2 c_1_2)
        (adjacent c_4_4 c_4_5)
        (adjacent c_5_0 c_4_0)
        (adjacent c_5_1 c_6_1)
        (adjacent c_3_4 c_3_5)
        (adjacent c_0_4 c_0_3)
        (adjacent c_1_6 c_0_6)
        (adjacent c_4_3 c_3_3)
        (adjacent c_3_4 c_2_4)
        (adjacent c_5_6 c_4_6)
        (adjacent c_0_5 c_1_5)
        (adjacent c_2_4 c_1_4)
        (adjacent c_4_2 c_4_1)
        (adjacent c_4_5 c_4_4)
        (adjacent c_0_5 c_0_4)
        (adjacent c_4_6 c_4_5)
        (adjacent c_7_2 c_6_2)
        (adjacent c_0_1 c_0_0)
        (adjacent c_4_4 c_5_4)
        (adjacent c_3_5 c_4_5)
        (adjacent c_2_3 c_1_3)
        (at p1 c_4_2)
        (at t1 c_3_6)
        (empty t1)
    )

    (:goal
        (at p1 c_1_2)
    )

    
    
    
)

