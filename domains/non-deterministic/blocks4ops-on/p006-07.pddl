(define (problem blocks-006-07)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5)
    (:init (clear b3) (clear b5) (handempty) (on b0 b4) (on b1 b0) (on b3 b1) (on b4 b2) (ontable b2) (ontable b5))
    (:goal (on b5 b0))
)