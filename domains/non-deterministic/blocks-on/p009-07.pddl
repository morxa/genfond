(define (problem blocks-009-07)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8)
    (:init (clear b3) (handempty) (on b1 b8) (on b2 b7) (on b3 b1) (on b4 b6) (on b5 b4) (on b6 b2) (on b7 b0) (on b8 b5) (ontable b0))
    (:goal (on b7 b5))
)