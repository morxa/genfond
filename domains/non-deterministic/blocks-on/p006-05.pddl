(define (problem blocks-006-05)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5)
    (:init (clear b0) (clear b4) (handempty) (on b0 b3) (on b2 b1) (on b4 b5) (on b5 b2) (ontable b1) (ontable b3))
    (:goal (on b0 b2))
)