(define (problem problem-11-1)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b11 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b1) (clear b8) (clear b9) (on b10 b5) (on b2 b6) (on b3 b2) (on b4 b3) (on b5 b7) (on b6 b11) (on b7 b4) (on b9 b10) (on-table b1) (on-table b11) (on-table b8))
    (:goal (and (on b1 b6) (on b2 b1) (on b5 b2) (on b6 b4) (on b7 b9) (on b8 b5) (on b9 b8)))
)