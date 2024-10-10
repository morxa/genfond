
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; instance file automatically generated by the tarski fstrips writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem instance-10-3)
    (:domain blocksworld-atomic)

    (:objects
        b1 b10 b2 b3 b4 b5 b6 b7 b8 b9 - object
    )

    (:init
        (on b1 table)
        (on b6 b5)
        (on b4 b3)
        (on b2 b9)
        (on b10 b8)
        (on b8 b7)
        (on b3 b10)
        (on b5 b1)
        (on b7 table)
        (on b9 table)
        (clear b2)
        (clear b4)
        (clear b6)
        (clear table)
        (diff b5 b3)
        (diff b5 b4)
        (diff b2 b7)
        (diff b6 b7)
        (diff b7 b6)
        (diff b8 b9)
        (diff b9 b5)
        (diff table b2)
        (diff b10 b6)
        (diff b2 b8)
        (diff b10 b3)
        (diff b7 b3)
        (diff b6 b10)
        (diff b3 b9)
        (diff table b5)
        (diff b1 b10)
        (diff b2 b5)
        (diff table b8)
        (diff b5 b6)
        (diff b1 b5)
        (diff b1 table)
        (diff b3 b4)
        (diff b8 b10)
        (diff b10 b8)
        (diff b3 b1)
        (diff table b7)
        (diff b8 b2)
        (diff b6 b4)
        (diff b7 table)
        (diff b4 b10)
        (diff b9 b2)
        (diff b1 b4)
        (diff b1 b2)
        (diff b3 table)
        (diff b1 b3)
        (diff b6 b9)
        (diff b3 b10)
        (diff b8 b4)
        (diff b4 b5)
        (diff b7 b1)
        (diff b8 b1)
        (diff b7 b5)
        (diff b7 b4)
        (diff b9 b6)
        (diff b2 b6)
        (diff b3 b7)
        (diff b9 b4)
        (diff b4 b2)
        (diff b8 b5)
        (diff b1 b9)
        (diff b8 table)
        (diff b7 b2)
        (diff table b6)
        (diff b5 b8)
        (diff b7 b10)
        (diff b6 b5)
        (diff b9 b7)
        (diff b2 b9)
        (diff b8 b7)
        (diff b4 b9)
        (diff b9 b10)
        (diff b5 b7)
        (diff b3 b5)
        (diff b4 b6)
        (diff b2 b4)
        (diff b2 b1)
        (diff b5 b1)
        (diff b3 b2)
        (diff b10 b9)
        (diff b2 table)
        (diff b8 b6)
        (diff b2 b10)
        (diff b8 b3)
        (diff b6 b2)
        (diff b6 b8)
        (diff b3 b6)
        (diff b7 b9)
        (diff b1 b8)
        (diff b4 b1)
        (diff b9 table)
        (diff table b10)
        (diff b4 b7)
        (diff b5 b9)
        (diff b4 b3)
        (diff b6 b1)
        (diff b10 b2)
        (diff b4 b8)
        (diff table b3)
        (diff table b4)
        (diff b3 b8)
        (diff table b1)
        (diff b10 b4)
        (diff b1 b7)
        (diff b4 table)
        (diff table b9)
        (diff b10 b1)
        (diff b5 b2)
        (diff b2 b3)
        (diff b9 b3)
        (diff b5 table)
        (diff b10 b5)
        (diff b9 b1)
        (diff b10 table)
        (diff b1 b6)
        (diff b6 b3)
        (diff b5 b10)
        (diff b7 b8)
        (diff b6 table)
        (diff b10 b7)
        (diff b9 b8)
    )

    (:goal
        (and (on b9 table) (on b8 b9) (on b3 table) (on b7 table) (on b2 b8) (on b1 b2) (on b6 b1) (on b4 b7) (on b10 table) (on b5 b6))
    )

    
    
    
)

