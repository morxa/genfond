(define (problem blocks-018-09)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b14) (clear b15) (clear b16) (handempty) (on b0 b8) (on b10 b11) (on b11 b9) (on b12 b6) (on b13 b4) (on b14 b12) (on b15 b7) (on b16 b2) (on b17 b1) (on b2 b13) (on b3 b10) (on b4 b5) (on b6 b0) (on b8 b3) (on b9 b17) (ontable b1) (ontable b5) (ontable b7))
    (:goal (on b14 b5))
)