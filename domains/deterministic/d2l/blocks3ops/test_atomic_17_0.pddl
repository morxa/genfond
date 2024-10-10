
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; instance file automatically generated by the tarski fstrips writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem instance-17-0)
    (:domain blocksworld-atomic)

    (:objects
        b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9 - object
    )

    (:init
        (on b15 table)
        (on b11 table)
        (on b8 b11)
        (on b16 b15)
        (on b5 table)
        (on b13 table)
        (on b6 b16)
        (on b10 b7)
        (on b17 b13)
        (on b7 b9)
        (on b1 b8)
        (on b12 b14)
        (on b9 table)
        (on b2 b17)
        (on b3 b2)
        (on b14 table)
        (on b4 table)
        (clear b4)
        (clear b1)
        (clear b12)
        (clear b10)
        (clear b6)
        (clear table)
        (clear b5)
        (clear b3)
        (diff b5 b3)
        (diff b1 b15)
        (diff b11 b15)
        (diff b5 b4)
        (diff b7 b13)
        (diff b2 b7)
        (diff b14 b2)
        (diff b6 b7)
        (diff b7 b6)
        (diff b8 b9)
        (diff b9 b5)
        (diff b10 b16)
        (diff table b2)
        (diff b7 b16)
        (diff b10 b6)
        (diff b5 b15)
        (diff b13 b16)
        (diff b12 b15)
        (diff b15 b3)
        (diff b16 b2)
        (diff b17 b5)
        (diff b17 b12)
        (diff b2 b8)
        (diff b12 b10)
        (diff b15 b6)
        (diff b10 b3)
        (diff table b14)
        (diff b1 b13)
        (diff b3 b17)
        (diff b11 b13)
        (diff b15 b16)
        (diff b12 b7)
        (diff b7 b3)
        (diff b14 table)
        (diff b12 table)
        (diff b14 b5)
        (diff b6 b10)
        (diff b10 b14)
        (diff b6 b13)
        (diff b3 b9)
        (diff table b5)
        (diff b1 b16)
        (diff b11 b16)
        (diff b17 b2)
        (diff b1 b10)
        (diff b11 b10)
        (diff b2 b5)
        (diff table b8)
        (diff b5 b6)
        (diff b16 b13)
        (diff table b16)
        (diff b1 b5)
        (diff b11 b5)
        (diff b17 b8)
        (diff b12 b4)
        (diff b1 table)
        (diff b3 b4)
        (diff b10 b13)
        (diff b11 table)
        (diff b6 b14)
        (diff b13 b8)
        (diff b8 b10)
        (diff b10 b8)
        (diff b14 b8)
        (diff b14 b13)
        (diff b16 b3)
        (diff b8 b17)
        (diff b3 b1)
        (diff table b7)
        (diff b3 b11)
        (diff b8 b2)
        (diff b12 b16)
        (diff b4 b13)
        (diff b6 b4)
        (diff b7 table)
        (diff b12 b6)
        (diff b14 b3)
        (diff b4 b10)
        (diff b9 b2)
        (diff b5 b14)
        (diff b17 b11)
        (diff b2 b13)
        (diff b14 b7)
        (diff b16 b5)
        (diff b1 b4)
        (diff b1 b2)
        (diff b3 table)
        (diff b11 b2)
        (diff b11 b4)
        (diff b1 b3)
        (diff b6 b9)
        (diff b6 b16)
        (diff b11 b3)
        (diff b13 table)
        (diff b15 b1)
        (diff b15 b11)
        (diff b3 b10)
        (diff b8 b4)
        (diff b4 b5)
        (diff b14 b15)
        (diff b3 b15)
        (diff b15 b7)
        (diff b7 b1)
        (diff b7 b11)
        (diff b10 b12)
        (diff b13 b12)
        (diff b8 b1)
        (diff b8 b11)
        (diff b9 b12)
        (diff b12 b2)
        (diff b12 b9)
        (diff b7 b5)
        (diff b7 b4)
        (diff b16 b15)
        (diff b9 b6)
        (diff b15 b8)
        (diff b17 b6)
        (diff b16 b10)
        (diff b2 b6)
        (diff b13 b3)
        (diff b3 b7)
        (diff b2 b16)
        (diff b6 b15)
        (diff b9 b13)
        (diff b16 b7)
        (diff b17 b3)
        (diff b16 table)
        (diff b9 b4)
        (diff b4 b2)
        (diff b8 b5)
        (diff b1 b12)
        (diff b1 b9)
        (diff b11 b9)
        (diff b8 table)
        (diff b11 b12)
        (diff b7 b2)
        (diff b15 b14)
        (diff b4 b14)
        (diff b16 b1)
        (diff table b6)
        (diff b5 b8)
        (diff b16 b11)
        (diff b12 b13)
        (diff b7 b10)
        (diff table b12)
        (diff b16 b4)
        (diff b6 b5)
        (diff b9 b7)
        (diff b2 b9)
        (diff b14 b9)
        (diff b14 b12)
        (diff b8 b7)
        (diff b4 b12)
        (diff b4 b9)
        (diff b9 b10)
        (diff b5 b7)
        (diff b3 b5)
        (diff b10 b15)
        (diff b4 b6)
        (diff b13 b15)
        (diff b13 b17)
        (diff b17 table)
        (diff b9 b15)
        (diff b14 b17)
        (diff b2 b4)
        (diff b12 b5)
        (diff b7 b17)
        (diff b17 b7)
        (diff b2 b1)
        (diff b2 b11)
        (diff b6 b12)
        (diff b5 b1)
        (diff b5 b11)
        (diff b1 b17)
        (diff b3 b13)
        (diff b3 b2)
        (diff b11 b17)
        (diff b15 b17)
        (diff b8 b16)
        (diff b10 b9)
        (diff b2 table)
        (diff b8 b6)
        (diff b9 b16)
        (diff b16 b9)
        (diff b17 b4)
        (diff b2 b10)
        (diff b12 b8)
        (diff b8 b3)
        (diff b2 b12)
        (diff b6 b2)
        (diff b16 b17)
        (diff b4 b15)
        (diff b15 b2)
        (diff b6 b8)
        (diff b8 b14)
        (diff b13 b5)
        (diff b9 b14)
        (diff b3 b6)
        (diff b2 b14)
        (diff b7 b9)
        (diff b1 b8)
        (diff b3 b16)
        (diff b4 b1)
        (diff b4 b11)
        (diff b7 b15)
        (diff b9 table)
        (diff table b10)
        (diff b11 b8)
        (diff b12 b1)
        (diff b4 b7)
        (diff b12 b11)
        (diff b5 b9)
        (diff b5 b12)
        (diff b17 b10)
        (diff b7 b14)
        (diff b8 b13)
        (diff b14 b10)
        (diff b9 b17)
        (diff b4 b3)
        (diff b6 b1)
        (diff b6 b11)
        (diff b13 b10)
        (diff b15 b9)
        (diff b14 b16)
        (diff b10 b17)
        (diff b16 b12)
        (diff b4 b16)
        (diff b13 b7)
        (diff b10 b2)
        (diff b5 b17)
        (diff b13 b2)
        (diff b4 b8)
        (diff b1 b11)
        (diff b11 b1)
        (diff b17 b14)
        (diff b15 b4)
        (diff table b3)
        (diff b14 b1)
        (diff b13 b14)
        (diff b14 b11)
        (diff table b4)
        (diff b15 table)
        (diff b17 b13)
        (diff b3 b8)
        (diff b14 b4)
        (diff b13 b4)
        (diff table b1)
        (diff table b11)
        (diff b10 b4)
        (diff b1 b7)
        (diff b2 b17)
        (diff b11 b7)
        (diff b4 table)
        (diff b6 b17)
        (diff b8 b12)
        (diff b13 b1)
        (diff table b9)
        (diff table b15)
        (diff b7 b12)
        (diff b10 b1)
        (diff b5 b2)
        (diff b10 b11)
        (diff b13 b6)
        (diff b2 b3)
        (diff b13 b11)
        (diff b17 b16)
        (diff b9 b3)
        (diff b15 b10)
        (diff table b13)
        (diff b1 b14)
        (diff b11 b14)
        (diff b16 b8)
        (diff b15 b12)
        (diff b16 b14)
        (diff b12 b17)
        (diff b17 b1)
        (diff b5 table)
        (diff b10 b5)
        (diff b9 b1)
        (diff b9 b11)
        (diff b10 table)
        (diff b1 b6)
        (diff b11 b6)
        (diff b17 b9)
        (diff b12 b14)
        (diff b3 b12)
        (diff b6 b3)
        (diff b5 b10)
        (diff b15 b5)
        (diff b17 b15)
        (diff b7 b8)
        (diff b4 b17)
        (diff b6 table)
        (diff b13 b9)
        (diff b5 b16)
        (diff b14 b6)
        (diff b12 b3)
        (diff b10 b7)
        (diff table b17)
        (diff b3 b14)
        (diff b8 b15)
        (diff b9 b8)
        (diff b5 b13)
        (diff b16 b6)
        (diff b15 b13)
        (diff b2 b15)
    )

    (:goal
        (and (on b16 table) (on b10 b16) (on b15 table) (on b5 b15) (on b6 table) (on b2 b5) (on b11 b6) (on b9 b11) (on b13 b2) (on b1 b9) (on b3 b10) (on b7 b1) (on b14 b3) (on b12 table) (on b17 b14) (on b8 b13) (on b4 b8))
    )

    
    
    
)

