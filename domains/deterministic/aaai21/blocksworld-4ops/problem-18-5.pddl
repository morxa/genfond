(define (problem problem-18-5)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b17) (clear b2) (clear b4) (clear b9) (on b1 b11) (on b11 b10) (on b13 b6) (on b14 b5) (on b15 b16) (on b17 b18) (on b18 b14) (on b2 b8) (on b3 b13) (on b4 b7) (on b5 b3) (on b6 b1) (on b7 b15) (on b9 b12) (on-table b10) (on-table b12) (on-table b16) (on-table b8))
    (:goal (and (on b1 b4) (on b2 b12) (on b3 b7) (on b4 b11) (on b5 b1) (on b8 b18) (on b9 b14) (on b11 b6) (on b13 b2) (on b14 b16) (on b15 b5) (on b17 b9) (on b18 b3)))
)