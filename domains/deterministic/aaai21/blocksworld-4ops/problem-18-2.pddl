(define (problem problem-18-2)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b12) (clear b15) (clear b18) (clear b2) (clear b7) (on b10 b17) (on b11 b1) (on b12 b3) (on b13 b5) (on b14 b13) (on b16 b4) (on b18 b11) (on b4 b8) (on b5 b9) (on b6 b14) (on b7 b16) (on b8 b6) (on b9 b10) (on-table b1) (on-table b15) (on-table b17) (on-table b2) (on-table b3))
    (:goal (and (on b1 b5) (on b4 b7) (on b5 b13) (on b6 b1) (on b8 b12) (on b9 b11) (on b11 b2) (on b12 b18) (on b15 b10) (on b16 b4) (on b17 b14) (on b18 b6)))
)