(define (problem problem-15-5)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b11 b12 b13 b14 b15 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b1) (clear b2) (clear b6) (clear b8) (on b1 b4) (on b12 b11) (on b13 b15) (on b14 b7) (on b15 b14) (on b4 b5) (on b5 b10) (on b6 b13) (on b7 b12) (on b8 b9) (on b9 b3) (on-table b10) (on-table b11) (on-table b2) (on-table b3))
    (:goal (and (on b1 b5) (on b3 b10) (on b5 b6) (on b6 b7) (on b8 b1) (on b10 b2) (on b12 b15) (on b15 b11)))
)