(define (problem problem-6-4)
    (:domain blocksworld-3ops)
    (:objects b1 b2 b3 b4 b5 b6)
    (:init (arm-empty) (clear b3) (clear b6) (on b1 b4) (on b3 b5) (on b4 b2) (on b6 b1) (on-table b2) (on-table b5))
    (:goal (and (on b2 b1) (on b4 b3)))
)