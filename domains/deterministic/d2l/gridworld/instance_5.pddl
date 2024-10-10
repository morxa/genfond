
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Instance file automatically generated by the Tarski FSTRIPS writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem gridworld-5x5)
    (:domain gridworld)

    (:objects
        
    )

    (:init
        (= (xpos ) 1)
        (= (ypos ) 1)
        (= (maxpos ) 5)
        (= (goal_xpos ) 5)
        (= (goal_ypos ) 5)
    )

    (:goal
        (and (= (xpos ) (goal_xpos )) (= (ypos ) (goal_ypos )))
    )

    
    (:bounds
        (coordinate - int[1..5]))
    
)

