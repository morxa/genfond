
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Instance file automatically generated by the Tarski FSTRIPS writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem delivery_6x1_p-3)
    (:domain delivery)

    (:objects
        c_0_0 c_1_0 c_2_0 c_3_0 c_4_0 c_5_0 - cell
        p1 p2 p3 - package
        t1 - truck
    )

    (:init
        (adjacent c_2_0 c_1_0)
        (adjacent c_3_0 c_2_0)
        (adjacent c_4_0 c_3_0)
        (adjacent c_1_0 c_2_0)
        (adjacent c_2_0 c_3_0)
        (adjacent c_1_0 c_0_0)
        (adjacent c_3_0 c_4_0)
        (adjacent c_5_0 c_4_0)
        (adjacent c_0_0 c_1_0)
        (adjacent c_4_0 c_5_0)
        (at p3 c_0_0)
        (at p1 c_1_0)
        (at t1 c_5_0)
        (at p2 c_3_0)
        (empty t1)
    )

    (:goal
        (and (at p1 c_2_0) (at p2 c_2_0) (at p3 c_2_0))
    )

    
    
    
)

