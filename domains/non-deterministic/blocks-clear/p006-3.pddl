(define (problem blocks-006-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5)
    (:init (clear b0) (clear b1) (handempty) (on b0 b4) (on b1 b3) (on b3 b5) (on b5 b2) (ontable b2) (ontable b4))
    (:goal (clear b3))
)