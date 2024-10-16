
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Instance file automatically generated by the Tarski FSTRIPS writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem p5x5_0)
    (:domain reward-strips)

    (:objects
        c_0_0 c_0_1 c_0_2 c_0_3 c_0_4 c_1_0 c_1_1 c_1_2 c_1_3 c_1_4 c_2_0 c_2_1 c_2_2 c_2_3 c_2_4 c_3_0 c_3_1 c_3_2 c_3_3 c_3_4 c_4_0 c_4_1 c_4_2 c_4_3 c_4_4 - cell
    )

    (:init
        (adjacent c_2_2 c_1_2)
        (adjacent c_3_2 c_2_2)
        (adjacent c_0_2 c_0_1)
        (adjacent c_1_3 c_1_2)
        (adjacent c_3_0 c_3_1)
        (adjacent c_4_3 c_4_4)
        (adjacent c_3_2 c_3_1)
        (adjacent c_0_1 c_0_0)
        (adjacent c_1_3 c_1_4)
        (adjacent c_0_1 c_0_2)
        (adjacent c_2_1 c_2_0)
        (adjacent c_1_0 c_0_0)
        (adjacent c_1_4 c_2_4)
        (adjacent c_0_0 c_1_0)
        (adjacent c_1_3 c_0_3)
        (adjacent c_2_2 c_2_3)
        (adjacent c_2_3 c_2_2)
        (adjacent c_1_0 c_1_1)
        (adjacent c_2_0 c_1_0)
        (adjacent c_0_3 c_1_3)
        (adjacent c_3_0 c_4_0)
        (adjacent c_1_1 c_2_1)
        (adjacent c_4_0 c_3_0)
        (adjacent c_3_3 c_3_2)
        (adjacent c_3_2 c_4_2)
        (adjacent c_3_4 c_3_3)
        (adjacent c_1_2 c_2_2)
        (adjacent c_3_0 c_2_0)
        (adjacent c_3_3 c_3_4)
        (adjacent c_2_4 c_2_3)
        (adjacent c_2_1 c_3_1)
        (adjacent c_3_3 c_4_3)
        (adjacent c_2_0 c_3_0)
        (adjacent c_3_1 c_4_1)
        (adjacent c_4_1 c_3_1)
        (adjacent c_1_1 c_1_0)
        (adjacent c_3_4 c_2_4)
        (adjacent c_1_1 c_0_1)
        (adjacent c_4_0 c_4_1)
        (adjacent c_1_2 c_1_1)
        (adjacent c_3_3 c_2_3)
        (adjacent c_3_2 c_3_3)
        (adjacent c_4_3 c_3_3)
        (adjacent c_0_0 c_0_1)
        (adjacent c_0_4 c_0_3)
        (adjacent c_3_1 c_3_2)
        (adjacent c_4_2 c_4_1)
        (adjacent c_1_4 c_0_4)
        (adjacent c_1_3 c_2_3)
        (adjacent c_2_3 c_2_4)
        (adjacent c_0_1 c_1_1)
        (adjacent c_2_2 c_2_1)
        (adjacent c_4_1 c_4_0)
        (adjacent c_0_2 c_1_2)
        (adjacent c_1_0 c_2_0)
        (adjacent c_2_3 c_1_3)
        (adjacent c_0_4 c_1_4)
        (adjacent c_2_1 c_2_2)
        (adjacent c_2_3 c_3_3)
        (adjacent c_4_2 c_3_2)
        (adjacent c_1_4 c_1_3)
        (adjacent c_3_1 c_2_1)
        (adjacent c_4_4 c_4_3)
        (adjacent c_4_2 c_4_3)
        (adjacent c_0_2 c_0_3)
        (adjacent c_0_3 c_0_2)
        (adjacent c_1_2 c_1_3)
        (adjacent c_3_4 c_4_4)
        (adjacent c_4_3 c_4_2)
        (adjacent c_4_1 c_4_2)
        (adjacent c_0_3 c_0_4)
        (adjacent c_1_1 c_1_2)
        (adjacent c_2_1 c_1_1)
        (adjacent c_4_4 c_3_4)
        (adjacent c_3_1 c_3_0)
        (adjacent c_2_4 c_1_4)
        (adjacent c_2_2 c_3_2)
        (adjacent c_1_2 c_0_2)
        (adjacent c_2_0 c_2_1)
        (adjacent c_2_4 c_3_4)
        (at c_0_0)
        (unblocked c_1_4)
        (unblocked c_0_4)
        (unblocked c_1_1)
        (unblocked c_4_1)
        (unblocked c_1_3)
        (unblocked c_3_2)
        (unblocked c_2_3)
        (unblocked c_3_4)
        (unblocked c_4_0)
        (unblocked c_0_1)
        (unblocked c_1_0)
        (unblocked c_3_0)
        (unblocked c_4_4)
        (unblocked c_3_3)
        (unblocked c_3_1)
        (unblocked c_0_0)
        (unblocked c_4_3)
        (unblocked c_4_2)
        (unblocked c_1_2)
        (unblocked c_2_4)
        (unblocked c_0_2)
        (unblocked c_2_2)
        (reward c_1_0)
        (reward c_3_0)
        (reward c_4_3)
    )

    (:goal
        (and (picked c_4_3) (picked c_3_0) (picked c_1_0))
    )

    
    
    
)

