(define (problem blocks-011-1)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b3) (handempty) (on b0 b2) (on b10 b0) (on b2 b5) (on b3 b8) (on b4 b7) (on b5 b6) (on b6 b4) (on b7 b9) (on b8 b10) (on b9 b1) (ontable b1))
    (:goal (clear b10))
)