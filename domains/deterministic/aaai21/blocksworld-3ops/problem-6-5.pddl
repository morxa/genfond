(define (problem problem-6-5)
    (:domain blocksworld-3ops)
    (:objects b1 b2 b3 b4 b5 b6)
    (:init (arm-empty) (clear b2) (clear b3) (on b1 b5) (on b2 b1) (on b3 b4) (on b4 b6) (on-table b5) (on-table b6))
    (:goal (and (on b4 b1) (on b5 b6)))
)