(define (problem blocks-011-05)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b3) (handempty) (on b1 b4) (on b10 b2) (on b2 b9) (on b3 b10) (on b4 b5) (on b6 b7) (on b7 b8) (on b8 b1) (on b9 b6) (ontable b0) (ontable b5))
    (:goal (on b8 b9))
)