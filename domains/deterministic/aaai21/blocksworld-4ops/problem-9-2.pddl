(define (problem problem-9-2)
    (:domain blocksworld-4ops)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b4) (clear b6) (on b1 b9) (on b2 b3) (on b3 b5) (on b4 b1) (on b6 b8) (on b8 b7) (on b9 b2) (on-table b5) (on-table b7))
    (:goal (and (on b1 b3) (on b2 b1) (on b4 b2) (on b5 b4) (on b6 b7) (on b7 b8) (on b8 b9) (on b9 b5)))
)