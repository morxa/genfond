(define (problem problem-17-4)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b1) (clear b17) (clear b2) (clear b3) (clear b6) (clear b7) (on b1 b14) (on b10 b11) (on b11 b5) (on b13 b8) (on b14 b13) (on b17 b12) (on b4 b9) (on b5 b4) (on b6 b10) (on b8 b16) (on b9 b15) (on-table b12) (on-table b15) (on-table b16) (on-table b2) (on-table b3) (on-table b7))
    (:goal (and (on b1 b6) (on b2 b12) (on b3 b2) (on b4 b14) (on b5 b16) (on b7 b3) (on b8 b5) (on b9 b15) (on b10 b8) (on b13 b17) (on b14 b13) (on b15 b10) (on b16 b4) (on b17 b1)))
)